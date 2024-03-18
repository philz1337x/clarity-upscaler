<div align="center">

<h1> Clarity-Upscaler: Reimagined Image Upscaling for Everyone </h1>

[![Website](https://img.shields.io/badge/Website-ClarityAI.cc-blueviolet)](https://ClarityAI.cc) [![Twitter Follow](https://img.shields.io/twitter/follow/philz1337x?style=social)](https://twitter.com/philz1337x)

[![Replicate](https://img.shields.io/badge/Demo-Replicate-purple)](https://replicate.com/philz1337x/clarity-upscaler)
![GitHub stars](https://img.shields.io/github/stars/philz1337x/clarity-upscaler?style=social&label=Star)

![Example video](example.gif)

<strong>Full Video:
https://x.com/philz1337x/status/1768679154726359128?s=20.</strong>

</div>

# 👋 Hello

I build open source AI apps. To finance my work I also build paid versions of my code. But feel free to use the free code. I post features and new projects on https://twitter.com/philz1337x

# 🚀 Options To Use Clarity-Upscaler

## User Friendly Website

If you are not familiar with cog, a1111 and don't want to use Replicate (which is quite simple), you can use my paid version at [ClarityAI.cc](https://ClarityAI.cc)

## Deploy and Run With cog (Locally or Cloud)

If you are not familiar with cog read: <a href=https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md>cog docs</a>

- Download Checkpoints and LoRa's from Cvitai and put in /models folder
- Predict with cog:

```su
cog predict -i image="link-to-image"
```

## Run With A1111 webUI

https://github.com/AUTOMATIC1111/stable-diffusion-webui

- Use these params:

```Prompt:
masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1> Negative prompt: (worst quality, low quality, normal quality:2) JuggernautNegative-neg Steps: 18, Sampler: DPM++ 3M SDE Karras, CFG scale: 6.0, Seed: 1337, Size: 1024x1024, Model hash: 338b85bc4f, Model: juggernaut_reborn, Denoising strength: 0.35, Tiled Diffusion upscaler: 4x-UltraSharp, Tiled Diffusion scale factor: 2, Tiled Diffusion: {"Method": "MultiDiffusion", "Tile tile width": 112, "Tile tile height": 144, "Tile Overlap": 4, "Tile batch size": 8, "Upscaler": "4x-UltraSharp", "Upscale factor": 2, "Keep input size": true}, ControlNet 0: "Module: tile_resample, Model: control_v11f1e_sd15_tile, Weight: 0.6, Resize Mode: 1, Low Vram: False, Processor Res: 512, Threshold A: 1, Threshold B: 1, Guidance Start: 0.0, Guidance End: 1.0, Pixel Perfect: True, Control Mode: 1, Hr Option: HiResFixOption.BOTH, Save Detected Map: False", Lora hashes: "more_details: 3b8aa1d351ef, SDXLrender_v2.0: 3925cf4759af"
```

## Replicate API for App Integration

- Go to https://replicate.com/philz1337x/clarity-upscaler/api
