import torchvision
torchvision.disable_beta_transforms_warning = lambda: None
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import time
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler
from diffusers.models import AutoencoderKL
from PIL import Image
import cv2
import numpy as np
import random
import math
import sys
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from torchvision import transforms
import pywt


# Load environment variables from .env file
load_dotenv()

# Ensure all paths are set correctly
controlnet_path_1 = os.getenv("CONTROLNET_MODEL_PATH_1")
controlnet_path_2 = os.getenv("CONTROLNET_MODEL_PATH_2")
sd_model_path = os.getenv("SD_MODEL_PATH")
vae_path = os.getenv("VAE_MODEL_PATH")
lora_weights1_path = os.getenv("LORA_WEIGHTS_PATH_1")
lora_weights2_path = os.getenv("LORA_WEIGHTS_PATH_2")
upscale_model = os.getenv("UPSCALE_MODEL")

# Define upscaler model paths
upscalers = {
    "4x_NMKD-Siax_200k": upscale_model,
}


if not all([controlnet_path_1, controlnet_path_2, sd_model_path, vae_path, lora_weights1_path, lora_weights2_path]):
    raise ValueError("Missing required environment variables in .env file")

USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def timer_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_scheduler(config):
    return LCMScheduler.from_config(config)

class LazyLoadPipeline:
    def __init__(self):
        self.pipe = None

    @timer_func
    def load(self):
        if self.pipe is None:
            print("Starting to load the pipeline...")
            self.pipe = self.setup_pipeline()
            print(f"Moving pipeline to device: {device}")
            self.pipe.to(device)
            if USE_TORCH_COMPILE:
                print("Compiling the model...")
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

    @timer_func
    def setup_pipeline(self):
        print("Setting up the pipeline...")
        controlnets = [
            ControlNetModel.from_single_file(
                controlnet_path_1, torch_dtype=torch.float16,
            ),
            ControlNetModel.from_single_file(
                controlnet_path_2, torch_dtype=torch.float16,
            ),
        ] # Load multiple controlnets       
        model_path = sd_model_path

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            model_path,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            control_guidance_end=[0.5, 1.0]
        )

        vae = AutoencoderKL.from_single_file(
            vae_path,
            torch_dtype=torch.float16
        )
        vae.enable_tiling()
        pipe.vae = vae
        # LCM_LoRA_Weights_SD15.safetensors
        pipe.load_lora_weights(lora_weights1_path)
        pipe.fuse_lora(lora_scale=1.0)
        #mode_details.safetensors
        pipe.load_lora_weights(lora_weights2_path)
        pipe.fuse_lora(lora_scale=0.25)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        return pipe

    def set_scheduler(self):
        if self.pipe is not None:
            self.pipe.scheduler = get_scheduler(self.pipe.scheduler.config)

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)
    
    
class UpscalerModel(torch.nn.Module):
    def __init__(self, scale=2.0):  # Add self parameter and default value
        super(UpscalerModel, self).__init__()  # Call parent class init first
        self.scale = scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to device once during initialization

    def forward(self, x):
        # Define the forward pass of your model
        return x

    def predict(self, image, upscale_model_path):
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Define transformation (resize the image for processing)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(image.height * self.scale), int(image.width * self.scale))),
            transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
        ])

        try:
            image_tensor = transform(image).to(self.device)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to transform image: {e}")

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(upscale_model_path)

        # Optionally, remove a prefix from keys if needed
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value

        try:
            self.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")

        self.eval()

        with torch.no_grad():
            upscaled_tensor = self(image_tensor)

        upscaled_image = upscaled_tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
        upscaled_image = Image.fromarray(upscaled_image.astype(np.uint8))
        return upscaled_image


def create_hdr_effect(original_image, hdr):
    """
    Applies an HDR effect to the given image.

    Args:
        original_image (PIL.Image.Image): The original image to which the HDR effect will be applied.
        hdr (float): The intensity of the HDR effect. A higher value results in a stronger effect.

    Returns:
        PIL.Image.Image: The image with the HDR effect applied.
    """    
    cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    factors = [1.0 - 0.9 * hdr, 1.0 - 0.7 * hdr, 1.0 - 0.45 * hdr,
              1.0 - 0.25 * hdr, 1.0, 1.0 + 0.2 * hdr,
              1.0 + 0.4 * hdr, 1.0 + 0.6 * hdr, 1.0 + 0.8 * hdr]
    images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    return Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))

lazy_pipe = LazyLoadPipeline()
lazy_pipe.load()

def wavelet_color_transfer(img1, img2, wavelet='haar', level=2):
    """
    Transfers color from img1 (reference) to img2 using wavelet-based processing.
    
    Args:
        img1: Reference image (color source).
        img2: Target image (to be color-corrected).
        wavelet: Type of wavelet transform ('haar', 'db2', etc.).
        level: Number of decomposition levels.

    Returns:
        Color-corrected image (img2 with img1's color tones).
    """
    # Convert images to LAB color space
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l1, a1, b1 = cv2.split(img1_lab)
    l2, a2, b2 = cv2.split(img2_lab)

    # Define function for wavelet decomposition and reconstruction
    def wavelet_transfer(channel1, channel2):
        coeffs1 = pywt.wavedec2(channel1, wavelet, level=level)
        coeffs2 = pywt.wavedec2(channel2, wavelet, level=level)

        # Transfer A & B wavelet coefficients from image1 to image2
        coeffs2_transferred = list(coeffs2)
        coeffs2_transferred[0] = coeffs1[0]  # Replace Approximation coefficients

        # Reconstruct image2 channel with transferred coefficients
        return pywt.waverec2(coeffs2_transferred, wavelet)

    # Apply wavelet transfer on A & B channels (color information)
    a2_corrected = np.clip(wavelet_transfer(a1, a2), 0, 255).astype(np.uint8)
    b2_corrected = np.clip(wavelet_transfer(b1, b2), 0, 255).astype(np.uint8)

    # Merge L from img2 (preserve structure) with A & B from img1 (color transfer)
    corrected_lab = cv2.merge((l2, a2_corrected, b2_corrected))
    corrected_img = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return corrected_img


def upscale_image_with_model(input_image, scale_factor, model):
    """
    Upscale an image by a given scale factor using a specific upscaling model.
    
    Args:
        input_image: PIL Image or path to image file
        scale_factor: float, the desired upscaling factor
        model: UpscalerModel instance for the specific model
        Returns:
        PIL Image: The upscaled image
    """
    # Load image if path is provided
    if isinstance(input_image, str):
        input_image = Image.open(input_image)

    # Apply upscaling with selected model and scale factor
    # Create upscaler model with selected scale factor
    upscaler_model = UpscalerModel(scale_factor)
    
    try:
        # Get the path for the selected upscaler model
        upscaler_path = upscalers[model]
        # Upscale the image
        upscaled_image = upscaler_model.predict(input_image, upscaler_path)
    except Exception as e:
        raise RuntimeError(f"Error during upscaling: {e}")
    return upscaled_image


"""
The first function create_gaussian_weight creates a special weight pattern that looks like a bell curve (Gaussian distribution) across a square grid. 
It takes two inputs: the size of the square tile and a sigma value (defaulting to 0.3) that controls how quickly the weight falls off from the center.
The function creates a grid of x and y coordinates, then calculates weights that are highest in the middle and gradually decrease towards the edges.
This kind of weight pattern is often used in image processing to smoothly blend or transition between different areas.

The second function adaptive_tile_size helps determine appropriate dimensions for processing an image in tiles.
It takes an image size (width and height) and two optional parameters: a base tile size (default 512) and maximum tile size (default 1024).
The function calculates tile dimensions that maintain the original image's aspect ratio while staying within the size limits. It does this by:
1. Calculating the image's aspect ratio (width divided by height)
2. If the image is wider than tall, it starts with the width and calculates a proportional height
3. If the image is taller than wide, it starts with the height and calculates a proportional width
4. Makes sure the tiles aren't smaller than the base size or larger than the maximum size
The output is a tuple of two numbers representing the width and height of the tiles that should be used for processing the image.
This adaptive approach helps ensure efficient image processing by using appropriately sized tiles that maintain the image's proportions while staying within reasonable memory limits.

Both functions work together in a larger image processing system, where the gaussian weights might be used for blending tile edges,
and the adaptive tile sizes help manage memory usage when working with large images.
"""

def create_gaussian_weight(tile_size, sigma=0.3):
    x = np.linspace(-1, 1, tile_size)
    y = np.linspace(-1, 1, tile_size)
    xx, yy = np.meshgrid(x, y)
    gaussian_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian_weight

def adaptive_tile_size(image_size, base_tile_size=512, max_tile_size=1024):
    w, h = image_size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        tile_w = min(w, max_tile_size)
        tile_h = min(int(tile_w / aspect_ratio), max_tile_size)
    else:
        tile_h = min(h, max_tile_size)
        tile_w = min(int(tile_h * aspect_ratio), max_tile_size)
    return max(tile_w, base_tile_size), max(tile_h, base_tile_size)


"""
This function processes a single tile (portion) of an image using AI image generation techniques.

Purpose: The function takes an image tile and enhances it using AI models (specifically two "controlnets") with some guidance parameters to improve its quality.

Inputs:
- tile: A piece of an image to process
- num_inference_steps: How many times the AI should refine the image
- strength: How much the AI should modify the original image
- guidance_scale: How closely the AI should follow the given prompts

Outputs:
- Returns a numpy array containing the processed image tile

How it works:
First, it sets up two text prompts - one positive ("masterpiece, best quality, highres") telling the AI what to aim for, and one negative ("low quality, ugly, blurry...") telling it what to avoid.
The function then prepares the tile for processing by duplicating it into a list of two copies - one for each controlnet AI model that will process it.

It creates an options dictionary containing all the settings for the AI processing, including:
- The prompts
- The original image tile
- The duplicated tiles for control
- Various parameters that control the processing
- A random seed generator for consistent but unique results

Finally, it runs the image through the AI pipeline (lazy_pipe) with these options and converts the result to a numpy array.
The key transformation happening here is that each tile gets enhanced by AI models that try to improve its quality while maintaining the original content, guided by the positive and negative prompts.
"""
def process_tile(tile, num_inference_steps, strength, guidance_scale):
    prompt = "masterpiece, best quality, highres"
    negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality"
    
    # Convert tile to a list for both image and control_image
    if isinstance(tile, Image.Image):
        tile_list = [tile] * 2  # Duplicate the tile for both controlnets
    
    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": tile,  # Original image
        "control_image": tile_list,  # List of control images for both controlnets
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": [1.0, 0.55],
        "control_guidance_end": [0.5, 1.0],
        "generator": torch.Generator(device=device).manual_seed(random.randint(0, 2147483647)),
    }
    
    return np.array(lazy_pipe(**options).images[0])

@timer_func
def process_image(input_image, scale_by, num_inference_steps, strength, hdr, guidance_scale):
    print("Starting image processing...")
    torch.cuda.empty_cache()
    lazy_pipe.set_scheduler()
    
    # Convert input_image to PIL Image if it's a path
    if isinstance(input_image, str):
        input_image = Image.open(input_image)
    
    # Convert input_image to numpy array
    input_array = np.array(input_image)
    
    # Prepare the condition image
    condition_image = upscale_image_with_model(input_image, scale_by, "4x_NMKD-Siax_200k")
    if hdr > 0.0:
        condition_image = create_hdr_effect(condition_image)
    
    condition_image_numpy = np.array(condition_image)
    W, H = condition_image.size

    # Adaptive tiling
    tile_width, tile_height = adaptive_tile_size((W, H))
    
    # Calculate the number of tiles
    overlap = min(64, tile_width // 8, tile_height // 8)  # Adaptive overlap
    num_tiles_x = math.ceil((W - overlap) / (tile_width - overlap))
    num_tiles_y = math.ceil((H - overlap) / (tile_height - overlap))
    
    # Create a blank canvas for the result
    result = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W, 1), dtype=np.float32)
    
    # Create gaussian weight
    gaussian_weight = create_gaussian_weight(max(tile_width, tile_height))

    num_inference_steps = int(num_inference_steps / strength)
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Calculate tile coordinates
            left = j * (tile_width - overlap)
            top = i * (tile_height - overlap)
            right = min(left + tile_width, W)
            bottom = min(top + tile_height, H)
            
            # Adjust tile size if it's at the edge
            current_tile_size = (bottom - top, right - left)
            
            tile = condition_image.crop((left, top, right, bottom))
            tile = tile.resize((tile_width, tile_height))
            
            # Process the tile
            result_tile = process_tile(tile, num_inference_steps, strength, guidance_scale)
            
            # Apply gaussian weighting
            if current_tile_size != (tile_width, tile_height):
                result_tile = cv2.resize(result_tile, current_tile_size[::-1])
                tile_weight = cv2.resize(gaussian_weight, current_tile_size[::-1])
            else:
                tile_weight = gaussian_weight[:current_tile_size[0], :current_tile_size[1]]
            
            # Add the tile to the result with gaussian weighting
            result[top:bottom, left:right] += result_tile * tile_weight[:, :, np.newaxis]
            weight_sum[top:bottom, left:right] += tile_weight[:, :, np.newaxis]
    
    # Normalize the result
    final_result = (result / weight_sum).astype(np.uint8)
    print("Image processing completed successfully")
    wavelet_image=wavelet_color_transfer(condition_image_numpy, final_result)
    print ("Wavelet color transfer completed successfully")
    return [input_array, wavelet_image]


def main():
    # Create an argument parser object with a description of your program.
    parser = argparse.ArgumentParser(description='Script for image generation')

    # Create mutual exclusion between input_image and input_directory
    group = parser.add_mutually_exclusive_group(required=True)

    # Add the input_image parameter which is now part of the mutually exclusive group
    group.add_argument('-i', '--input_image', help='Path to a single input image.')

    # Also add the input_directory parameter to this same mutual exclusion group 
    group.add_argument('-d', '--input_directory',
        help='Path to an directory containing multiple input images or directories.')    

    # Add the output_directory parameter which is optional
    parser.add_argument('-o', '--output_directory',
        default=None,
        help='Optional: Path to where the upscaled images should be saved. If not set, defaults will depend on whether an image or a directory was provided.')

    # Add other parameters as optional arguments that take default values if not specified.
    parser.add_argument('-r', '--scale_by', type=int, default=2.0,
                        help='Scale factor (2.0 or 4.0) of generated image (default: 2.0)')
    
    parser.add_argument('-n', '--num_inference_steps', type=int, default=20,
                        help='Number of inference steps for generation (default: 20)')

    parser.add_argument('-s', '--strength', type=float, default=1.0,
                        help='Strength parameter for the generation process (default: 1.0)')
    
    parser.add_argument('--hdr', type=float, default=0.0,
                        help='HDR Effect on generated image (default: 0)')

    parser.add_argument('-g', '--guidance_scale', type=float, default=3.0,
                        help='Guidance scale for the generation process (default: 3)')

    # Parse the command line arguments
    args = parser.parse_args()

    if args.output_directory:
        output_dir = args.output_directory
    elif args.input_image:
        # Check if output_directory is specified, else use the current directory
        if args.output_directory:
            output_dir = args.output_directory
        else:
            output_dir = '.'  # Use the current directory
    else:  # If input directory was used and no specific output dir given
        output_dir = os.path.join(args.input_directory, "UPSCALED")
        # Create the directory if it doesn't exist

    # Create 'upscaled_' prefix for the output image file name
    # base_path, ext = os.path.splitext(os.path.basename(args.input_image))
    # Now you can access your variables like this:
    input_image = args.input_image
    input_directory = args.input_directory
    scale_by = args.scale_by
    num_inference_steps = args.num_inference_steps
    strength = args.strength
    hdr = args.hdr
    guidance_scale = args.guidance_scale

        # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to process either a single image or all images in the directory
    def process_images(input_image, input_directory, scale_by, num_inference_steps, strength, hdr, guidance_scale, output_dir):
        if input_image:
            # Process single image
            base_path, ext = os.path.splitext(os.path.basename(input_image))
            input_image_path = input_image
            _, final_result = process_image(input_image_path, scale_by, num_inference_steps, strength, 
                                            hdr, guidance_scale)
            final_image = Image.fromarray(final_result)
            output_file = os.path.join(output_dir, f"upscaled_{base_path}{ext}")
            final_image.save(output_file)

        elif input_directory:
            # Process all images in the input directory
            files = os.listdir(input_directory)  # get list of files in the directory
            pbar = tqdm(files, desc="Processing images", total=len(files), unit="file")
            for file_name in pbar:
                if file_name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):  # Add more formats as needed
                    input_image_path = os.path.join(input_directory, file_name)
                    base_path, ext = os.path.splitext(file_name)
                    output_file = os.path.join(output_dir, f"upscaled_{base_path}{ext}")
                    # Update tqdm description with the current filename
                    pbar.set_description(f"Processing {file_name}")  # Use the pbar object to update the description
                    if os.path.exists(output_file):
                        print(f"Skipping {file_name}: already exists in output directory.")  # Debugging line  y
                    else:
                        _, final_result = process_image(input_image_path, scale_by, num_inference_steps, 
                                                        strength, hdr, guidance_scale)
                        final_image = Image.fromarray(final_result)
                        final_image.save(output_file)
                    pbar.update(1)  # Increment the progress bar
        else:
            # If neither input_image nor input_directory is set, return error
            print("Error: No input image or directory provided.")
            print("Usage: python samothraki_upscaler.py --input_image <image_path> OR --input_directory <directory_path>")
            sys.exit(1)

    # Call the process_images function and pass in the required arguments
    process_images(input_image, input_directory, scale_by, num_inference_steps, strength, hdr, guidance_scale, output_dir)

if __name__ == "__main__":
    main()
