# Custom Stable Diffusion Pipeline with Karras Scheduler

This project rebuilds the Stable Diffusion v1.5 inference pipeline from scratch,
integrating **Karras noise scheduling** to improve image quality and sampling efficiency.

### Features
- Pure PyTorch implementation (no Diffusers dependency)
- DDPM sampler with **optional Karras timesteps (ρ=7.0)**
- Supports both text-to-image and image-to-image
- Generates comparable results with 30–35 steps vs. 50 linear DDPM steps

### Project Structure
project/
├── sd/ # core pipeline
├── data/
├── images/
├── output/
├── v1-5-pruned-emaonly.ckpt
├── run.py
└── requirements.txt

### Usage
```bash
pip install -r requirements.txt
```
run demo.ipynb