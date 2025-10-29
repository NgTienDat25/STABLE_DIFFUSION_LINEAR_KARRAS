# Custom Stable Diffusion Pipeline with Karras Scheduler

This project rebuilds the Stable Diffusion v1.5 inference pipeline from scratch,
integrating **Karras noise scheduling** to improve image quality and sampling efficiency.

### Features
- Pure PyTorch implementation (no Diffusers dependency)
- DDPM sampler with **optional Karras timesteps (ρ=7.0)**
- Supports both text-to-image and image-to-image
- Generates comparable results with 30–35 steps vs. 50 linear DDPM steps


### Scheduler Comparison (Same Seed/Prompt)

<table align="center">
  <tr>
    <td align="center"><strong>Output with Linear Scheduler</strong></td>
    <td align="center"><strong>Output with Karras Scheduler</strong></td>
  </tr>
  <tr>
    <td>
      <img src="output/output_with_linear.png" alt="dog wearing glasses (Linear)" width="400">
    </td>
    <td>
      <img src="output/output_with_karras.png" alt="dog wearing glasses (Karras)" width="400">
    </td>
  </tr>
</table>

### Project Structure
```bash

project/
├── sd/ # core pipeline
├── data/
├── images/
├── output/
├── v1-5-pruned-emaonly.ckpt
├── demo.ipynb
└── requirements.txt
```
### Usage
```bash
pip install -r requirements.txt
```
run demo.ipynb