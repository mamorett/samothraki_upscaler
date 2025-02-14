# Samothraki Upscaler Documentation

The Samothraki Upscaler is a command-line tool designed for enhancing image quality using advanced upsampling techniques based on Stable Diffusion models.Processes either a single image or all images in the provided input directory, upscaling them and saving the results to the specified output directory.
    
    Args:
        input_image (str): Path to the input image to be upscaled.
        input_directory (str): Path to the directory containing the images to be upscaled.
        scale_by (int): The scale factor to use for upscaling the images (2.0 or 4.0).
        num_inference_steps (int): The number of inference steps to use for the upscaling process.
        strength (float): The strength parameter for the upscaling process.
        hdr (float): The HDR effect to apply to the generated images.
        guidance_scale (float): The guidance scale for the upscaling process.
        output_dir (str): The path to the output directory where the upscaled images will be saved.


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

### Output 

- Single image mode: Creates an upscaled version with "upscaled_" prefix
- Directory mode: Creates an "UPSCALED" subdirectory containing all processed images

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