"""
voxel_app.py

Gradio front-end for the text-to-voxel pipeline: type a prompt, see the
result as a rotatable 3D mesh in-browser, and download a .vox file to open
in MagicaVoxel for nicer shading/screenshots.

Rebuilds the model classes standalone (this file has no dependency on the
notebook's kernel state) and loads the already-trained weights:
    special_vae_shapenet_with_batchnorm.pth   -- frozen VAE encoder/decoder
    unet_syn_final.pth                        -- diffusion UNet (+ EMA weights)

Install:
    pip install gradio trimesh voxypy

Run:
    python voxel_app.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from diffusers import DDIMScheduler
from scipy import ndimage
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as hf_logging
from voxypy.models import Entity

import gradio as gr

hf_logging.set_verbosity_error()

# ---- Config ----

# resolve relative to this file, not the shell's cwd -- running via a full
# path (e.g. `python "C:/.../voxel_app.py"`) does not cd into this folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VAE_PATH = os.path.join(SCRIPT_DIR, "special_vae_shapenet_with_batchnorm.pth")
UNET_PATH = os.path.join(SCRIPT_DIR, "unet_syn_final.pth")
MAX_LEN = 8
CONTEXT_DIM = 512
UNCOND_PROMPT = ""
TIME_DIM = 128


# ---- VAE (must match the architecture the checkpoint was trained with) ----

class VoxelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 8, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class VoxelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class VoxelVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.voxel_encoder = VoxelEncoder()
        self.voxel_decoder = VoxelDecoder()

    def encode(self, x):
        x = self.voxel_encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        x = self.reparameterize(mu, logvar)
        return self.voxel_decoder(x), mu, logvar


# ---- UNet (must match the architecture unet_syn_final.pth was trained with) ----

def groups(ch):
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class CrossAttention(nn.Module):
    def __init__(self, channels, contextdim=CONTEXT_DIM, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, f"{channels} channels must divide {num_heads} heads"
        self.norm = nn.GroupNorm(groups(channels), channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(contextdim, channels)
        self.to_v = nn.Linear(contextdim, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x, context, contextmask=None):
        b, c, d, h, w = x.shape
        a = self.norm(x).view(b, c, d * h * w).permute(0, 2, 1)
        q, k, v = self.to_q(a), self.to_k(context), self.to_v(context)
        attn, _ = self.mha(q, k, v, key_padding_mask=contextmask)
        return x + self.to_out(attn).permute(0, 2, 1).view(b, c, d, h, w)


class ResBlock(nn.Module):
    def __init__(self, inch, outch, timedim=TIME_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups(inch), inch)
        self.conv1 = nn.Conv3d(inch, outch, kernel_size=3, padding=1)
        self.time_fc = nn.Linear(timedim, outch)
        self.norm2 = nn.GroupNorm(groups(outch), outch)
        self.conv2 = nn.Conv3d(outch, outch, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(inch, outch, kernel_size=1) if inch != outch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_fc(F.silu(t_emb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, base=32, num_heads=4, contextdim=CONTEXT_DIM):
        super().__init__()
        from diffusers.models.embeddings import Timesteps, TimestepEmbedding

        b1, b2 = base, base * 2

        self.time_proj = Timesteps(num_channels=TIME_DIM, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_emb = TimestepEmbedding(in_channels=TIME_DIM, time_embed_dim=TIME_DIM)

        self.conv_in = nn.Conv3d(in_channels, b1, kernel_size=3, padding=1)

        self.down1_res = ResBlock(b1, b1)
        self.down1_attn = CrossAttention(b1, contextdim, num_heads)
        self.downsample = nn.Conv3d(b1, b1, kernel_size=3, stride=2, padding=1)

        self.down2_res = ResBlock(b1, b2)
        self.down2_attn = CrossAttention(b2, contextdim, num_heads)

        self.mid_res1 = ResBlock(b2, b2)
        self.mid_attn = CrossAttention(b2, contextdim, num_heads)
        self.mid_res2 = ResBlock(b2, b2)

        self.up2_res = ResBlock(b2 * 2, b2)
        self.up2_attn = CrossAttention(b2, contextdim, num_heads)
        self.upsample = nn.ConvTranspose3d(b2, b1, kernel_size=4, stride=2, padding=1)

        self.up1_res = ResBlock(b1 * 2, b1)
        self.up1_attn = CrossAttention(b1, contextdim, num_heads)

        self.out_norm = nn.GroupNorm(groups(b1), b1)
        self.conv_out = nn.Conv3d(b1, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, context, contextmask=None):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device)
        t = t.to(x.device).reshape(-1)
        if t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        temb = self.t_emb(self.time_proj(t))

        h = self.conv_in(x)
        h = self.down1_attn(self.down1_res(h, temb), context, contextmask)
        skip1 = h

        h = self.downsample(h)
        h = self.down2_attn(self.down2_res(h, temb), context, contextmask)
        skip2 = h

        h = self.mid_res1(h, temb)
        h = self.mid_attn(h, context, contextmask)
        h = self.mid_res2(h, temb)

        h = self.up2_res(torch.cat([h, skip2], dim=1), temb)
        h = self.up2_attn(h, context, contextmask)

        h = self.upsample(h)
        h = self.up1_res(torch.cat([h, skip1], dim=1), temb)
        h = self.up1_attn(h, context, contextmask)

        return self.conv_out(F.silu(self.out_norm(h)))


# ---- Load everything once, at process start ----

print("Loading VAE...")
specialvae = VoxelVAE()
specialvae.load_state_dict(torch.load(VAE_PATH, map_location="cpu", weights_only=True))
specialvae.eval()
for p in specialvae.parameters():
    p.requires_grad = False

print("Loading UNet checkpoint...")
ckpt = torch.load(UNET_PATH, map_location="cpu", weights_only=True)
base = ckpt["base"]
latentscale = ckpt["LATENT_SCALE"]
emaunet = UNet(in_channels=4, base=base)
emaunet.load_state_dict(ckpt["ema"])
emaunet.eval()

print("Loading CLIP text encoder...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
cliptextmodel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
cliptextmodel.eval()
for p in cliptextmodel.parameters():
    p.requires_grad = False


@torch.no_grad()
def encodecaptions(captions):
    tok = tokenizer(captions, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")
    emb = cliptextmodel(**tok).last_hidden_state
    return emb, ~tok.attention_mask.bool()


captioncache = {}


def lookupcaptions(captions):
    missing = [c for c in dict.fromkeys(captions) if c not in captioncache]
    if missing:
        e, m = encodecaptions(missing)
        for i, c in enumerate(missing):
            captioncache[c] = (e[i], m[i])
    pairs = [captioncache[c] for c in captions]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


# ---- Figure out which captions the model was actually trained on ----

voxelroot = next(
    (d for d in (os.path.join(SCRIPT_DIR, "voxelized_shapenet"),
                 os.path.expanduser("~/Downloads/voxelized_shapenet")) if os.path.isdir(d)),
    os.path.join(SCRIPT_DIR, "voxelized_shapenet"),
)
manifestpath = os.path.join(voxelroot, "manifest.csv")
if os.path.exists(manifestpath):
    df = pd.read_csv(manifestpath).dropna(subset=["caption"])
    trainedcaptions = sorted(set(df["caption"].tolist()))
else:
    print(f"WARNING: no manifest.csv found at {manifestpath} -- falling back to the original 4 categories")
    trainedcaptions = ["a airplane", "a car", "a chair", "a table"]

print(f"Trained categories: {trainedcaptions}")

_, _ = lookupcaptions(trainedcaptions + [UNCOND_PROMPT])


@torch.no_grad()
def pooled(texts):
    tok = tokenizer(texts, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")
    return F.normalize(cliptextmodel(**tok).pooler_output, dim=-1)


trainedpooled = pooled(trainedcaptions)


@torch.no_grad()
def routeprompt(prompt, verbose=True):
    if prompt in trainedcaptions or prompt == UNCOND_PROMPT:
        return prompt
    sim = (pooled([prompt]) @ trainedpooled.T)[0]
    best = trainedcaptions[int(sim.argmax())]
    if verbose:
        print(f"  routed {prompt!r} -> {best!r}  (cos {sim.max():.3f})")
    return best


def largestcomponent(vox):
    lab, n = ndimage.label(vox)
    if n <= 1:
        return vox
    sizes = ndimage.sum(vox, lab, range(1, n + 1))
    return (lab == 1 + int(np.argmax(sizes))).astype(int)


@torch.no_grad()
def generate(prompt, guidancescale=3.0, numsteps=50, seed=None, route=True, clean=True):
    routedprompt = routeprompt(prompt) if route else prompt

    sampler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)
    sampler.set_timesteps(numsteps)

    g = torch.Generator().manual_seed(seed) if seed is not None else None
    latent = torch.randn((1, 4, 8, 8, 8), generator=g)

    cond, condmask = lookupcaptions([routedprompt])
    uncond, uncondmask = lookupcaptions([UNCOND_PROMPT])
    context = torch.cat([uncond, cond])
    mask = torch.cat([uncondmask, condmask])

    for t in sampler.timesteps:
        pred = emaunet(torch.cat([latent, latent]), t.repeat(2), context, mask)
        preduncond, predcond = pred.chunk(2)
        pred = preduncond + guidancescale * (predcond - preduncond)
        latent = sampler.step(pred, t, latent).prev_sample

    probs = torch.sigmoid(specialvae.voxel_decoder(latent / latentscale))
    voxels = (probs > 0.5).squeeze().cpu().numpy().astype(int)

    if clean:
        voxels = largestcomponent(voxels)

    return voxels, routedprompt


# ---- Gradio glue: run the pipeline, write a mesh (for in-browser preview) and a .vox (for MagicaVoxel) ----

NUM_OUTPUTS = 6


def run(prompt, guidance_scale, steps, use_seed, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")

    outdir = tempfile.mkdtemp(prefix="voxel_app_")

    obj_paths = [None] * NUM_OUTPUTS
    vox_paths = [None] * NUM_OUTPUTS
    occupied_counts = []
    routedprompt = None

    for i in range(NUM_OUTPUTS):
        # each of the 6 is its own independent sampling run -- if a seed is
        # fixed we still vary it per-slot so they aren't 6 copies of the same result
        seed_val = int(seed) + i if use_seed else None
        voxels, routedprompt = generate(prompt, guidancescale=guidance_scale, numsteps=int(steps), seed=seed_val)
        occupied = int(voxels.sum())
        occupied_counts.append(occupied)

        if occupied == 0:
            continue

        # as_boxes() emits one cube per occupied voxel -- unlike marching_cubes,
        # which fits a smooth iso-surface through them, this keeps the blocky
        # look MagicaVoxel shows.
        voxelgrid = trimesh.voxel.VoxelGrid(voxels.astype(bool))
        mesh = voxelgrid.as_boxes()
        mesh.visual.face_colors = [255, 205, 90, 255]  # bright warm color, not the dim default gray
        obj_path = os.path.join(outdir, f"preview_{i}.glb")
        mesh.export(obj_path)
        obj_paths[i] = obj_path

        vox_path = os.path.join(outdir, f"output_{i}.vox")
        Entity(data=voxels.astype(int)).save(vox_path)
        vox_paths[i] = vox_path

    counts_str = ", ".join(str(c) for c in occupied_counts)
    status = f"'{prompt}' -> routed to '{routedprompt}'  |  occupied voxels per output: {counts_str}"
    return (*obj_paths, *vox_paths, status)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="a chair", value="a chair"),
        gr.Slider(
            1.0, 10.0, value=3.0, step=0.5, label="Guidance scale",
            info=(
                "How strictly the model follows your prompt. Low = more random/creative "
                "shapes that may drift from what you typed. High = sticks closely to the "
                "prompt but can look distorted or overcooked if pushed too far. Comes from "
                "comparing a 'with prompt' prediction against a 'no prompt' prediction at "
                "each denoising step and exaggerating the difference between them."
            ),
        ),
        gr.Slider(
            10, 100, value=50, step=5, label="Sampling steps",
            info=(
                "How many denoising passes the model takes to turn random noise into a "
                "finished voxel shape. Few steps = faster but rougher/less detailed. Many "
                "steps = slower but cleaner and more refined, with diminishing returns past "
                "a point. Each step nudges the noisy voxel grid a bit closer to a real shape."
            ),
        ),
        gr.Checkbox(label="Use fixed seed", value=False),
        gr.Number(label="Seed", value=0, precision=0),
    ],
    outputs=(
        [gr.Model3D(label=f"Preview {i + 1} (drag to rotate)", clear_color=(0.9, 0.9, 0.9, 1.0))
         for i in range(NUM_OUTPUTS)]
        + [gr.File(label=f"Download .vox {i + 1} (open in MagicaVoxel)") for i in range(NUM_OUTPUTS)]
        + [gr.Textbox(label="Status", interactive=False)]
    ),
    title="Text → Voxel",
    description=(
        f"Trained categories: {', '.join(trainedcaptions)}. "
        "Anything else gets routed to the closest trained category via CLIP similarity. "
        f"Generates {NUM_OUTPUTS} independent samples per prompt."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
