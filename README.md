# AIO-Stuff

A personal workspace with two parts:

1. **[Text-to-Voxel Generator](#text-to-voxel-generator)** is a polished, runnable project: type a word and get 3D voxel models back.
2. **[Homework](#homework)** holds course notebooks and worksheets I use for practice. They are working files, not finished products.

---

## Text-to-Voxel Generator

Type a prompt like "a chair" and the generator produces six different 3D voxel models. You can rotate each one in the browser and download it as a `.vox` file to open in [MagicaVoxel](https://ephtracy.github.io/).

Built jointly with Caleb.

### How it works

- **3D VAE.** A convolutional variational autoencoder compresses a 32x32x32 voxel grid into a small latent volume (4 channels, 8x8x8). Its weights are frozen after training.
- **Latent diffusion.** A 3D U-Net with cross-attention learns to denoise that latent volume, conditioned on text.
- **CLIP text conditioning.** Prompts are embedded with CLIP (ViT-B/32), so the model learns a link between word embeddings and voxel structure rather than a fixed label list.
- **Sampling.** DDIM sampling with classifier-free guidance. Guidance scale and step count are adjustable in the app.
- **Prompt routing.** A prompt outside the trained categories is matched to the closest trained category by CLIP similarity.

The longer-term goal is a model that generalizes from word-model pairs instead of memorizing a few hardcoded words. With more training pairs, it may be able to do one-shot generalization. That is future work and is not claimed here.

### Run it

```bash
pip install torch diffusers transformers gradio trimesh voxypy scipy pandas numpy
python voxel_app.py
```

The trained weights are in this repo (`special_vae_shapenet_with_batchnorm.pth` and `unet_syn_final.pth`). The training dataset is not included, so the app uses its built-in default category list unless a `voxelized_shapenet/manifest.csv` sits next to the script.

### Files

| File | Purpose |
|---|---|
| `voxel_app.py` | Gradio app: model definitions, sampling, mesh preview and `.vox` export |
| `special_vae_shapenet_with_batchnorm.pth` | Trained VAE weights |
| `unet_syn_final.pth` | Trained diffusion U-Net weights (including EMA) |

---

## Homework

The `Homework/` folder contains Jupyter notebooks I work through for coursework and self-study:

- **Machine learning:** derivatives, linear algebra, basis expansion and regularization, backpropagation and gradient descent, optimizers, PyTorch classification, autoencoders and GANs.
- **Physics:** an AP Physics 1 course notebook.

Many of the worksheets were originally written by Cooper. Credit to him. These notebooks are practice files and are not part of the generator.
