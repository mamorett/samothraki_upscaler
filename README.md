# Samothraki Upscaler Documentation

The Samothraki Upscaler is a command-line tool designed for enhancing image quality using advanced upsampling techniques based on Stable Diffusion models.

## Quick Start Guide

### Prerequisites

Create a `.env` file containing the following variables:

```bash
CONTROLNET_MODEL_PATH_1="<your_dir>/models/controlnet/control_v11f1e_sd15_tile.safetensors"
CONTROLNET_MODEL_PATH_2="<your_dir>/models/controlnet/control_v11p_sd15_inpaint_fp16.safetensors"
SD_MODEL_PATH="/gorgon/ia/modelli/checkpoints/sd15/juggernaut_reborn.safetensors"
VAE_MODEL_PATH="<your_dir>/models/vae/vae-ft-mse-840000-ema-pruned.ckpt"
LORA_WEIGHTS_PATH_1="<your_dir>/models/loras/LCM_LoRA_Weights_SD15.safetensors" 
LORA_WEIGHTS_PATH_2="<your_dir>/models/loras/more_details.safetensors"
UPSCALE_MODEL="<your_dir>/models/upscale_models/4x_NMKD-Siax_200k.pth"
```

### Basic Usage
To use the Samothraki Upscaler, you can run:

```bash
python -m samothraki.upscaler [input] [-o output]
```

Where:
- `input` is either a single image file or a directory of input images.
- `-o output` specifies the output directory (optional).

### Required Parameters

#### Input Parameters
You must provide exactly one of the following:

1. **Single Image**
```bash
samothraki.upscaler -i input_image.jpg
```

2. **Directory of Images**
```bash
samothraki.upscaler -d input_directory/
```

### Optional Parameters

- `--scale_by`: Scale factor for upsampling (defaults to 2.0). Use values like `4.0` for higher magnifications.
  
  ```bash
  samothraki.upscaler -i input.jpg --scale_by 4.0
  ```

- `--num_inference_steps`: Number of sampling steps for image generation (defaults to 20).

  ```bash
  samothraki.upscaler -d input_dir --num_inference_steps 30
  ```

- `--strength`: Strength parameter for upscaling effects (defaults to 1.0).

  ```bash
  samothraki.upscaler -i image.jpg --strength 0.8
  ```

### Output Directory

If the output directory is not specified, images will be saved in a subdirectory named `upscaled/` by default.

### Environment Variables

The tool requires specific paths to model files:

1. **ControlNet Models**
```text
CONTROLNET_MODEL_PATH_1="/gorgon/ia/ComfyUI/models/controlnet/control_v11f1e_sd15_tile.safetensors"
CONTROLNET_MODEL_PATH_2="/gorgon/ia/ComfyUI/models/controlnet/control_v11p_sd15_inpaint_fp16.safetensors"
```

2. **Stable Diffusion Model**
```text
SD_MODEL_PATH="/gorgon/ia/modelli/checkpoints/sd15/juggernaut_reborn.safetensors"
```

3. **VAE Model**
```text
VAE_MODEL_PATH="/gorgon/ia/ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.ckpt"
```

4. **LoRA Weights (Optional)**
```text
LORA_WEIGHTS_PATH_1="/gorgon/ia/ComfyUI/models/loras/LCM_LoRA_Weights_SD15.safetensors"
LORA_WEIGHTS_PATH_2="/gorgon/ia/ComflyUI/models/loras/more_details.safetensors"
```

5. **Upscale Model**
```text
UPSCALE_MODEL="/gorgon/ia/ComfyUI/models/upscale_models/4x_NMKD-Samothraki-128.safetensors"
```

## Customization

If you wish to adjust parameters, consider using the following flags:

- `--batch_size`: Number of images to process simultaneously (defaults to 1).
  
  ```bash
  samothraki.upscaler -d input_dir --batch_size 4
  ```

- `--continue_from`: Resume processing from a specific output folder.

  ```bash
  samothraki.upscaler -i resume_from/0002.jpg --continue_from resume_from/
  ```

## Examples

1. Upscale a single image with custom settings:

   ```bash
   samothraki.upscaler -i example.jpg --scale_by 4.0 --strength 0.7
   ```

2. Process an entire directory of images:

   ```bash
   samothraki.upscaler -d input_dir --num_inference_steps 35 --output_dir output/
   ```

## Notes

- Ensure all required environment variables are set before running the tool.
  
- Adjust parameters according to your specific needs for image quality and processing speed.

For detailed instructions, refer to the official documentation or use `python -h` within the module for help messages.
```

This guide provides a comprehensive overview of how to use the Samothraki Upscaler, including command-line options, required parameters, optional configurations, and environment variables.