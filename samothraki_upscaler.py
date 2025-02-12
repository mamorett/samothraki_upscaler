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
            safety_checker=None
        )

        vae = AutoencoderKL.from_single_file(
            vae_path,
            torch_dtype=torch.float16
        )
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
    def __init__(self, scale):
        self.scale = scale
        super(UpscalerModel, self).__init__()
        # Define your layers here, based on the architecture you expect
        # This is just a placeholder for demonstration purposes
        pass
    
    def forward(self, x):
        # Define the forward pass of your model
        return x

    def predict(self, image):
        # Load the image
        # image = Image.open(input_image_path).convert("RGB")

        # Define transformation (resize the image for processing)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image.height * self.scale, image.width * self.scale)),  # Upscale 2x
            transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
        ])

        # Apply transformation
        image_tensor = transform(image).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Initialize the model
        model = UpscalerModel(scale=self.scale).to(image_tensor.device)

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(upscale_model)

        # Optionally, remove a prefix from keys if needed (as before)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")  # Adjust the prefix removal based on the error message
            new_state_dict[new_key] = value

        # Load the state dictionary with strict=False to allow mismatched keys
        try:
            model.load_state_dict(new_state_dict, strict=False)  # Ignore missing/unexpected keys
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")

        # Set the model to evaluation mode
        model.eval()

        # Perform inference to upscale the image
        with torch.no_grad():
            upscaled_tensor = model(image_tensor)

        # Convert the tensor back to an image
        upscaled_image = upscaled_tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
        upscaled_image = Image.fromarray(upscaled_image.astype(np.uint8))
        return upscaled_image


NMKD_Upscaler_x2 = UpscalerModel(scale=2)
NMKD_Upscaler_x4 = UpscalerModel(scale=4)

@timer_func
def resize_and_upscale(input_image, resolution):
    if isinstance(input_image, str):
        input_image = Image.open(input_image)  # Load the image from the file path     
    scale = 2 if resolution <= 2048 else 4
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H = int(round(H * k / 64.0)) * 64
    W = int(round(W * k / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    if scale == 2:
        img = NMKD_Upscaler_x2.predict(img)
    else:
        img = NMKD_Upscaler_x4.predict(img)
    return img

@timer_func
def create_hdr_effect(original_image, hdr):
    if hdr == 0:
        return original_image
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


@timer_func
def progressive_upscale(input_image, target_resolution, steps=3):
    if isinstance(input_image, str):
        input_image = Image.open(input_image)  # Load the image from the file path    
    current_image = input_image.convert("RGB")
    current_size = max(current_image.size)
    
    for _ in range(steps):
        if current_size >= target_resolution:
            break
        
        scale_factor = min(2, target_resolution / current_size)
        new_size = (int(current_image.width * scale_factor), int(current_image.height * scale_factor))
        
        if scale_factor <= 2.0:
            current_image = NMKD_Upscaler_x2.predict(current_image)
        else:
            current_image = NMKD_Upscaler_x4.predict(current_image)
        
        current_size = max(current_image.size)
    
    # Final resize to exact target resolution
    if current_size != target_resolution:
        aspect_ratio = current_image.width / current_image.height
        if current_image.width > current_image.height:
            new_size = (target_resolution, int(target_resolution / aspect_ratio))
        else:
            new_size = (int(target_resolution * aspect_ratio), target_resolution)
        current_image = current_image.resize(new_size, Image.LANCZOS)
    
    return current_image

def prepare_image(input_image, resolution, hdr):
    upscaled_image = progressive_upscale(input_image, resolution)
    return create_hdr_effect(upscaled_image, hdr)

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

def process_tile(tile, num_inference_steps, strength, guidance_scale):
    prompt = "masterpiece, best quality, highres"
    negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality, verybadimagenegative_v1.3, JuggernautNegative-neg"
    
    # Convert tile to a list for both image and control_image
    if isinstance(tile, Image.Image):
        tile_list = [tile] * 2  # Duplicate the tile for both controlnets
    else:
        tile_list = [tile] * 2
    
    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": tile,  # Original image
        "control_image": tile_list,  # List of control images for both controlnets
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": [1.0, 0.55],
        "generator": torch.Generator(device=device).manual_seed(random.randint(0, 2147483647)),
    }
    
    return np.array(lazy_pipe(**options).images[0])

@timer_func
def process_image(input_image, resolution, num_inference_steps, strength, hdr, guidance_scale):
    print("Starting image processing...")
    torch.cuda.empty_cache()
    lazy_pipe.set_scheduler()
    
    # Convert input_image to PIL Image if it's a path
    if isinstance(input_image, str):
        input_image = Image.open(input_image)
    
    # Convert input_image to numpy array
    input_array = np.array(input_image)
    
    # Prepare the condition image
    condition_image = prepare_image(input_image, resolution, hdr)
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
    parser.add_argument('-r', '--resolution', type=int, default=2048,
                        help='Resolution of generated image (default: 2048)')
    
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
    resolution = args.resolution
    num_inference_steps = args.num_inference_steps
    strength = args.strength
    hdr = args.hdr
    guidance_scale = args.guidance_scale

        # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to process either a single image or all images in the directory
    def process_images(input_image, input_directory, resolution, num_inference_steps, strength, hdr, guidance_scale, output_dir):
        if input_image:
            # Process single image
            base_path, ext = os.path.splitext(os.path.basename(input_image))
            input_image_path = input_image
            _, final_result = process_image(input_image_path, resolution, num_inference_steps, strength, 
                                            hdr, guidance_scale)
            final_image = Image.fromarray(final_result)
            output_file = os.path.join(output_dir, f"upscaled_{base_path}{ext}")
            final_image.save(output_file)

        elif input_directory:
            # Process all images in the input directory
            for file_name in os.listdir(input_directory):
                if file_name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):  # Add more formats as needed
                    input_image_path = os.path.join(input_directory, file_name)
                    base_path, ext = os.path.splitext(file_name)
                    _, final_result = process_image(input_image_path, resolution, num_inference_steps, 
                                                    strength, hdr, guidance_scale, )
                    final_image = Image.fromarray(final_result)
                    output_file = os.path.join(output_dir, f"upscaled_{base_path}{ext}")
                    final_image.save(output_file)
        else:
            # If neither input_image nor input_directory is set, return error
            print("Error: No input image or directory provided.")
            print("Usage: python script.py --input_image <image_path> OR --input_directory <directory_path>")
            sys.exit(1)

    # Call the process_images function and pass in the required arguments
    process_images(input_image, input_directory, resolution, num_inference_steps, strength, hdr, guidance_scale, output_dir)

if __name__ == "__main__":
    main()
