
from modules import timer
from modules import initialize_util
from modules import initialize
from modules.tiling.img_utils import convert_pil_img_to_binary, \
                                    convert_binary_img_to_pil, \
                                    convert_pil_img_to_base64, \
                                    convert_base64_img_to_pil, \
                                    shift_image, \
                                    draw_center_cross_image
from modules.debugging.debug_image import debug_tiling_image, \
                                          expand_canvas_tiling, \
                                          save_output_img
from handfix.handfix import (detect_and_crop_hand_from_binary, insert_cropped_hand_into_image)
from urllib.parse import urlparse
from fastapi import FastAPI
from io import BytesIO
from PIL import Image, ImageFilter

import os, json
import numpy as np
import requests
import base64
import uuid
import time
import cv2
import mimetypes
import subprocess
import logging

from cog import BasePredictor, Input, Path

mimetypes.add_type("image/webp", ".webp")

# Fixing the "DecompressionBombWarning" warning
Image.MAX_IMAGE_PIXELS = None

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        try:
            self.logger = logging.getLogger(__name__)
            os.environ['IGNORE_CMD_ARGS_ERRORS'] = '1'

            startup_timer = timer.startup_timer
            startup_timer.record("launcher")

            try:
                initialize.imports()
                initialize.check_versions()
                initialize.initialize()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize: {str(e)}")

            try:
                app = FastAPI()
                initialize_util.setup_middleware(app)

                from modules.api.api import Api
                from modules.call_queue import queue_lock

                self.api = Api(app, queue_lock)
            except Exception as e:
                raise RuntimeError(f"Failed to setup FastAPI and API: {str(e)}")

            try:
                model_response = self.api.get_sd_models()
                self.logger.info("Available checkpoints: %s", str(model_response))
            except Exception as e:
                raise RuntimeError(f"Failed to get SD models: {str(e)}")

            try:
                from modules import script_callbacks
                script_callbacks.before_ui_callback()
                script_callbacks.app_started_callback(None, app)

                from modules.api.models import StableDiffusionImg2ImgProcessingAPI
                self.StableDiffusionImg2ImgProcessingAPI = StableDiffusionImg2ImgProcessingAPI
            except Exception as e:
                raise RuntimeError(f"Failed to setup callbacks and processing API: {str(e)}")

            try:
                file_path = Path("init.png")
                base64_encoded_data = base64.b64encode(file_path.read_bytes())
                base64_image = base64_encoded_data.decode('utf-8')
            except FileNotFoundError:
                raise FileNotFoundError("init.png file not found")
            except Exception as e:
                raise RuntimeError(f"Failed to encode init image: {str(e)}")

        except Exception as e:
            self.logger.error("Critical error during setup: %s", str(e))
            raise

        try:
            payload = {
               "override_settings": {
                    "sd_model_checkpoint": "juggernaut_reborn.safetensors",
                    "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
                     "CLIP_stop_at_last_layers": 1,
                },
                "override_settings_restore_afterwards": False,
                "prompt": "office building",
                "steps": 1,
                "init_images": [base64_image],
                "denoising_strength": 0.1,
                "do_not_save_samples": True,
                "alwayson_scripts": {
                    "Tiled Diffusion": {
                        "args": [
                            True,
                            "MultiDiffusion",
                            True,
                            True,
                            1,
                            1,
                            112,
                            144,
                            4,
                            8,
                            "4x-UltraSharp",
                            1.1,
                            False,
                            0,
                            0.0,
                            3,
                        ]
                    },
                    "Tiled VAE": {
                        "args": [
                            True,
                            3072,
                            192,
                            True,
                            True,
                            True,
                            True,
                        ]

                    },
                    "controlnet": {
                        "args": [
                            {
                                "enabled": True,
                                "module": "tile_resample",
                                "model": "control_v11f1e_sd15_tile",
                                "weight": 0.2,
                                "image": base64_image,
                                "resize_mode": 1,
                                "lowvram": False,
                                "downsample": 1.0,
                                "guidance_start": 0.0,
                                "guidance_end": 1.0,
                                "control_mode": 1,
                                "pixel_perfect": True,
                                "threshold_a": 1,
                                "threshold_b": 1,
                                "save_detected_map": False,
                                "processor_res": 512,
                            }
                        ]
                    }
                }
            }

            try:
                req = StableDiffusionImg2ImgProcessingAPI(**payload)
            except Exception as e:
                raise RuntimeError(f"Failed to create processing request: {str(e)}")

            try:
                self.api.img2imgapi(req)
            except Exception as e:
                raise RuntimeError(f"Failed to process image: {str(e)}")

            self.logger.info("Startup time: %s", startup_timer.summary())

        except Exception as e:
            self.logger.error("Critical error during image processing setup and execution: %s", str(e))
            raise

    def download_lora_weights(self, url: str):
        folder_path = "models/Lora"

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        if "civitai.com" in parsed_url.netloc:
            filename = f"{os.path.basename(parsed_url.path)}.safetensors"

        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, filename)

        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as file:
            file.write(response.content)

        print("Lora saved under:", file_path)
        return file_path

    def download_safetensors(self, url: str):
        start_time_custom = time.time()

        safetensors_path = f"models/Stable-diffusion/custom-{uuid.uuid1()}.safetensors"

        subprocess.check_output(["pget", url, safetensors_path])

        print(f"Checkpoint downloading with pget took {round(time.time() - start_time_custom, 2)} seconds")

        return safetensors_path

    def calc_scale_factors(self, value):
        lst = []
        while value >= 2:
            lst.append(2)
            value /= 2
        if value > 1:
            lst.append(value)
        return lst

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description="Prompt", default="masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>"),
        negative_prompt: str = Input(description="Negative Prompt", default="(worst quality, low quality, normal quality:2) JuggernautNegative-neg"),
        scale_factor: float = Input(
            description="Scale factor", default=2
        ),
        dynamic: float = Input(
            description="HDR, try from 3 - 9", ge=1, le=50, default=6
        ),
        creativity: float = Input(
            description="Creativity, try from 0.3 - 0.9", ge=0, le=1, default=0.35
        ),
        resemblance: float = Input(
            description="Resemblance, try from 0.3 - 1.6", ge=0, le=3, default=0.6
        ),
        tiling_width: int = Input(
            description="Fractality, set lower tile width for a high Fractality",
            choices=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256],
            default=112
        ),
        tiling_height: int = Input(
            description="Fractality, set lower tile height for a high Fractality",
            choices=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256],
            default=144
        ),
        sd_model: str = Input(
            description="Stable Diffusion model checkpoint",
            choices=['epicrealism_naturalSinRC1VAE.safetensors [84d76a0328]', 'juggernaut_reborn.safetensors [338b85bc4f]', 'flat2DAnimerge_v45Sharp.safetensors'],
            default="juggernaut_reborn.safetensors [338b85bc4f]",
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a', 'Euler', 'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a', 'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras', 'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential', 'DPM fast', 'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras', 'Restart', 'DDIM', 'PLMS', 'UniPC'],
            default="DPM++ 3M SDE Karras",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=18
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=1337
        ),
        multistep_factor: float = Input(
            description="Multiplier for the number of denoising steps. 0.9 for 90% less steps, 1.1 for 10% more steps", ge=0, le=2, default=0.8
        ),
        downscaling: bool = Input(
            description="Downscale the image before upscaling. Can improve quality and speed for images with high resolution but lower quality", default=False
        ),
        downscaling_resolution: int = Input(
            description="Downscaling resolution", default=768
        ),
        lora_links: str = Input(
            description="Link to a lora file you want to use in your upscaling. Multiple links possible, seperated by comma",
            default=""
        ),
        custom_sd_model: str = Input(
            default=""
        ),
        sharpen: float = Input(
            description="Sharpen the image after upscaling. The higher the value, the more sharpening is applied. 0 for no sharpening", ge=0, le=10, default=0
        ),
        mask: Path = Input(
            description="Mask image to mark areas that should be preserved during upscaling", default=None
        ),
        handfix: str = Input(
            description="Use clarity to fix hands in the image",
            choices=['disabled', 'hands_only', 'image_and_hands'],
            default="disabled",
        ),
        pattern: bool = Input(
            description="Upscale a pattern with seamless tiling",
            default=False,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="png",
        )
    ) -> list[Path]:
        """Run a single prediction on the model"""
        # Start prediction process and record time
        self.logger.info("Starting prediction process")
        start_time = time.time()
        
        # print all the input parameters
        self.logger.info(f"Input parameters: {locals()}")

        # Initialize outputs list for storing intermediate results
        outputs = []

        try:
            # Handle model name corrections due to disabled hashing
            if sd_model == "epicrealism_naturalSinRC1VAE.safetensors [84d76a0328]":
                sd_model = "epicrealism_naturalSinRC1VAE.safetensors"
            if sd_model == "juggernaut_reborn.safetensors [338b85bc4f]":
                sd_model = "juggernaut_reborn.safetensors"

            # Download and process LoRA weights if provided
            if lora_links:
                try:
                    lora_link = [link.strip() for link in lora_links.split(",")]
                    for link in lora_link:
                        self.download_lora_weights(link)
                except Exception as e:
                    self.logger.error(f"Error downloading LoRA weights: {str(e)}")
                    raise

            # Handle custom SD model if provided
            if custom_sd_model:
                try:
                    path_to_custom_checkpoint = self.download_safetensors(custom_sd_model)
                    sd_model = os.path.basename(path_to_custom_checkpoint)
                    self.api.refresh_checkpoints()
                except Exception as e:
                    self.logger.error(f"Error processing custom SD model: {str(e)}")
                    raise

            # Process input image
            image_file_path = image
            try:
                with open(image_file_path, "rb") as image_file:
                    binary_image_data = image_file.read()
            except IOError as e:
                self.logger.error(f"Error reading input image: {str(e)}")
                raise

            # Handle mask if provided
            if mask:
                try:
                    with Image.open(image_file_path) as img:
                        original_resolution = img.size
                except Exception as e:
                    self.logger.error(f"Error processing mask: {str(e)}")
                    raise

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

        try:
            # Handle image downscaling if enabled
            if downscaling:
                try:
                    # Convert binary image data to numpy array
                    image_np_array = np.frombuffer(binary_image_data, dtype=np.uint8)
                    image = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
                    
                    # Get image dimensions
                    height, width = image.shape[:2]
                    
                    # Calculate scaling factor based on longest dimension
                    if height > width:
                        scaling_factor = downscaling_resolution / float(height)
                    else:
                        scaling_factor = downscaling_resolution / float(width)

                    # Calculate new dimensions
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)

                    # Resize image
                    resized_image = cv2.resize(image, (new_width, new_height))
                    
                    # Convert back to binary data
                    _, binary_resized_image = cv2.imencode('.jpg', resized_image)
                    binary_image_data = binary_resized_image.tobytes()
                    
                except Exception as e:
                    self.logger.error(f"Error during image downscaling: {str(e)}")
                    raise

            # Handle hand detection and fixing if enabled
            if handfix == "hands_only":
                try:
                    self.logger.info("Attempting to detect and fix hands in image")
                    binary_image_data_full_image = binary_image_data
                    cropped_hand_img, hand_coords = detect_and_crop_hand_from_binary(binary_image_data_full_image)
                    
                    if cropped_hand_img is not None:
                        self.logger.info("Hands detected in image - processing")
                        _, buffer = cv2.imencode('.jpg', cropped_hand_img)
                        binary_image_data = buffer.tobytes()

                        cropped_hand_img_rgb = cv2.cvtColor(cropped_hand_img, cv2.COLOR_BGR2RGB)
                        cropped_hand_img_pil = Image.fromarray(cropped_hand_img_rgb)
                    else:
                        self.logger.info("No hands detected in image")
                        return
                        
                except Exception as e:
                    self.logger.error(f"Error during hand detection/fixing: {str(e)}")
                    raise

            # Convert image to base64 for processing
            try:
                base64_encoded_data = base64.b64encode(binary_image_data)
                base64_image = base64_encoded_data.decode('utf-8')
            except Exception as e:
                self.logger.error(f"Error encoding image to base64: {str(e)}")
                raise

            # Calculate upscaling multipliers
            multipliers = [scale_factor]
            if scale_factor > 2:
                multipliers = self.calc_scale_factors(scale_factor)
                self.logger.info(f"Image will be upscaled {len(multipliers)} times")

            first_iteration = True

        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {str(e)}")
            raise
        try:
            # Iterate through each upscaling multiplier
            for i, multiplier in enumerate(multipliers):
                self.logger.info(f"Upscaling with scale_factor: {multiplier}")

                # Adjust creativity and seed for subsequent iterations
                if not first_iteration:
                    creativity = creativity * multistep_factor
                    seed = seed + 1

                first_iteration = False

                # Handle seamless tiling pattern if enabled
                if pattern:
                    try:
                        self.logger.info('Preparing seamless tiling process')
                        init_img = convert_base64_img_to_pil(base64_image)
                        # Expand canvas for tiling
                        expanded_img = expand_canvas_tiling(init_img, div=8, darken=False)
                        # Update base64 image with expanded canvas
                        base64_image = convert_pil_img_to_base64(expanded_img)
                    except Exception as e:
                        self.logger.error(f"Error in tiling pattern preparation: {str(e)}")
                        raise

                try:
                    # Generate payload and process image
                    payload = get_clarity_upscaler_payload(sd_model, tiling_width, tiling_height, multiplier, base64_image,
                                    resemblance, prompt, negative_prompt, num_inference_steps, dynamic, seed, scheduler,
                                    creativity)

                    req = self.StableDiffusionImg2ImgProcessingAPI(**payload)
                    resp = self.api.img2imgapi(req)
                    info = json.loads(resp.info)
                    base64_image = resp.images[0]
                except Exception as e:
                    self.logger.error(f"Error in image processing: {str(e)}")
                    raise

                # Handle seamless tiling post-processing
                if pattern:
                    try:
                        self.logger.info('Starting seamless tiling post-processing')
                        
                        # Convert and crop image
                        image_data = base64.b64decode(base64_image)
                        upscaled_img = Image.open(BytesIO(image_data))
                        width = upscaled_img.width
                        height = upscaled_img.height
                        border_size = int(width / 10)
                        cropped_back = upscaled_img.crop((border_size, border_size, width - border_size, height - border_size))

                        # First pass: Shift and process seams
                        shift_x = cropped_back.width // 2
                        shift_y = cropped_back.height // 2
                        seamless_tiling_overlap_width = 1.0
                        seamless_tiling_overlap_blur = 1.0
                        shifted_img_A = shift_image(cropped_back, shift_x, shift_y)
                        shifted_img_A_base64 = convert_pil_img_to_base64(shifted_img_A)
                        inpaint_mask_A_base64, inpaint_mask_A = get_seamless_tiling_mask(shifted_img_A_base64,
                                                                                seamless_tiling_overlap_width,
                                                                                seamless_tiling_overlap_blur)

                        # Process first pass
                        payload = get_clarity_upscaler_payload(sd_model, tiling_width, tiling_height, multiplier,
                                                            shifted_img_A_base64,
                                                            resemblance, prompt, negative_prompt, num_inference_steps,
                                                            dynamic, seed, scheduler, creativity,
                                                            seamfix_mask=inpaint_mask_A_base64)
                        req = self.StableDiffusionImg2ImgProcessingAPI(**payload)
                        resp = self.api.img2imgapi(req)
                        info = json.loads(resp.info)

                        # Process results of first pass
                        base64_image = resp.images[0]
                        gen_bytes = BytesIO(base64.b64decode(base64_image))
                        seam_fix_A = Image.open(gen_bytes)
                        shiftback_img_A = shift_image(seam_fix_A, -shift_x, -shift_y)

                        # Second pass: Process outer edges
                        shift_x = shiftback_img_A.width // 3
                        shift_y = shiftback_img_A.height // 3
                        shifted_img_B = shift_image(shiftback_img_A, shift_x, shift_y)
                        shifted_img_B_base64 = convert_pil_img_to_base64(shifted_img_B)

                        # Calculate offset parameters
                        fourth = (shifted_img_B.width // 4)
                        third = (shifted_img_B.width // 3)
                        fraction = (shifted_img_B.width // 20)
                        offset_x = (fourth + fourth) - (third + third)
                        offset_y = (fourth + fourth) - (third + third)

                        # Generate second mask
                        inpaint_mask_B_base64, inpaint_mask_B = get_seamless_tiling_mask(shifted_img_B_base64,
                                                                                seamless_tiling_overlap_width,
                                                                                seamless_tiling_overlap_blur * 1.2,
                                                                                offset_x=offset_x,
                                                                                offset_y=offset_y,
                                                                                x_start=fourth+fourth+third-fraction,
                                                                                x_end=fourth+fourth+third+fraction,
                                                                                y_start=fourth+fourth+third-fraction,
                                                                                y_end=fourth+fourth+third+fraction,
                                                                                boost=False)

                        # Final inpainting pass with reduced creativity
                        payload = get_clarity_upscaler_payload(sd_model, tiling_width, tiling_height, multiplier,
                                                            shifted_img_B_base64,
                                                            1.0, prompt, negative_prompt, num_inference_steps,
                                                            dynamic, seed, scheduler, 0.35,
                                                            seamfix_mask=inpaint_mask_B_base64)

                        req = self.StableDiffusionImg2ImgProcessingAPI(**payload)
                        resp = self.api.img2imgapi(req)
                        info = json.loads(resp.info)

                        # Process final results
                        base64_image = resp.images[0]
                        gen_bytes = BytesIO(base64.b64decode(base64_image))
                        seam_fix_B = Image.open(gen_bytes)
                        shiftback_img_B = shift_image(seam_fix_B, -shift_x, -shift_y)

                        # Convert final image back to base64
                        buffered = BytesIO()
                        shiftback_img_B.save(buffered, format="PNG")
                        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        resp.images[0] = base64_image

                    except Exception as e:
                        self.logger.error(f"Error in seamless tiling post-processing: {str(e)}")
                        raise

        except Exception as e:
            self.logger.error(f"Error in upscaling process: {str(e)}")
            raise


        # Process each generated image
        for i, image in enumerate(resp.images):
            try:
                # Get seed for this image
                seed = info.get("all_seeds", [])[i] or "unknown_seed"
                self.logger.info(f"Processing image {i+1} with seed {seed}")

                # Decode base64 image
                try:
                    gen_bytes = BytesIO(base64.b64decode(image))
                    imageObject = Image.open(gen_bytes)
                except Exception as e:
                    self.logger.error(f"Error decoding image: {str(e)}")
                    raise

                # Apply hand fixes if requested
                if handfix == "hands_only":
                    try:
                        imageObject = insert_cropped_hand_into_image(
                            binary_image_data_full_image, 
                            imageObject, 
                            hand_coords, 
                            cropped_hand_img_pil
                        )
                        self.logger.info("Applied hand fixes to image")
                    except Exception as e:
                        self.logger.error(f"Error applying hand fixes: {str(e)}")
                        raise

                # Apply mask if provided
                if mask:
                    try:
                        imageObject = imageObject.resize(original_resolution, Image.LANCZOS)
                        original_image = Image.open(image_file_path).resize(original_resolution, Image.LANCZOS)
                        mask_image = Image.open(mask).convert("L").resize(original_resolution, Image.LANCZOS)

                        # Apply gaussian blur to mask
                        blur_radius = 5
                        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
                        combined_image = Image.composite(original_image, imageObject, mask_image)

                        imageObject = combined_image
                        self.logger.info("Applied mask to image")
                    except Exception as e:
                        self.logger.error(f"Error applying mask: {str(e)}")
                        raise

                # Apply sharpening if requested
                if sharpen > 0:
                    try:
                        a = -sharpen / 10
                        b = 1 - 8 * a
                        kernel = [a, a, a, a, b, a, a, a, a]
                        kernel_filter = ImageFilter.Kernel((3, 3), kernel, scale=1, offset=0)
                        imageObject = imageObject.filter(kernel_filter)
                        self.logger.info(f"Applied sharpening with value {sharpen}")
                    except Exception as e:
                        self.logger.error(f"Error applying sharpening: {str(e)}")
                        raise

                # Save processed image
                try:
                    optimised_file_path = Path(f"{seed}-{uuid.uuid1()}.{output_format}")

                    if output_format in ["webp", "jpg"]:
                        imageObject.save(
                            optimised_file_path,
                            quality=95,
                            optimize=True,
                        )
                    else:
                        imageObject.save(optimised_file_path)
                    
                    outputs.append(optimised_file_path)
                    self.logger.info(f"Saved image as {optimised_file_path}")
                except Exception as e:
                    self.logger.error(f"Error saving image: {str(e)}")
                    raise

            except Exception as e:
                self.logger.error(f"Error processing image {i+1}: {str(e)}")
                raise

        # Cleanup custom model if used
        if custom_sd_model:
            try:
                os.remove(path_to_custom_checkpoint)
                self.logger.info(f"Removed custom checkpoint {path_to_custom_checkpoint}")
            except Exception as e:
                self.logger.error(f"Error removing custom checkpoint: {str(e)}")
                raise

        total_time = round(time.time() - start_time, 2)
        self.logger.info(f"Prediction completed in {total_time} seconds")
        return outputs

def get_seamless_tiling_mask(base64_image, width_mult, blur_mult,
                            offset_x=0, offset_y=0,
                            x_start=0, x_end=0, y_start=0, y_end=0,
                            boost=True):
    gen_bytes = BytesIO(base64.b64decode(base64_image))
    img = Image.open(gen_bytes)
    mask_pil = draw_center_cross_image(img, thickness_mult=width_mult, blur_mult=blur_mult,
                                    offset_x=offset_x, offset_y=offset_y,
                                    x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end,
                                    boost=boost)
    mask_base64 = convert_pil_img_to_base64(mask_pil)

    return mask_base64, mask_pil

def get_clarity_upscaler_payload(sd_model,
                                tiling_width,
                                tiling_height,
                                multiplier,
                                base64_image,
                                resemblance,
                                prompt,
                                negative_prompt,
                                num_inference_steps,
                                dynamic,
                                seed,
                                scheduler,
                                creativity,
                                seamfix_mask=None):
    if seamfix_mask:
        multiplier = 1.0 ## set the multiplier to 1 as we're not upscaling in this round.

    override_settings = {
        "sd_model_checkpoint": sd_model,
        "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
        "CLIP_stop_at_last_layers": 1,
    }
    alwayson_scripts = {
        "Tiled Diffusion": {"args": get_tiled_diffusion_args(tiling_width, tiling_height, multiplier)},
        "Tiled VAE": {"args": get_tiled_vae_args()},
        "controlnet": {"args": get_controlnet_args(base64_image, resemblance)}
    }

    if seamfix_mask:
        payload_dict = {
            "override_settings": override_settings,
            "override_settings_restore_afterwards": False,
            "init_images": [base64_image],
            "mask": seamfix_mask,
            "mask_blur": 0,
            "inpainting_fill": 1,  ## [fill, original, latent noise, latent nothing]
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 0,
            "inpainting_mask_invert": 0,
            "include_init_images": True,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "cfg_scale": dynamic,
            "seed": seed,
            "tiling": True,
            "do_not_save_samples": True,
            "sampler_name": scheduler,
            "denoising_strength": creativity,
            "alwayson_scripts": alwayson_scripts,
        }
    else:
        payload_dict = {
            "override_settings": override_settings,
            "override_settings_restore_afterwards": False,
            "init_images": [base64_image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "cfg_scale": dynamic,
            "seed": seed,
            "do_not_save_samples": True,
            "sampler_name": scheduler,
            "denoising_strength": creativity,
            "alwayson_scripts": alwayson_scripts,
        }

    return payload_dict

def get_tiled_diffusion_args(tiling_width, tiling_height, multiplier):
    arg_list = [
        True,
        "MultiDiffusion",
        True,
        True,
        1,
        1,
        tiling_width,
        tiling_height,
        4,
        8,
        "4x-UltraSharp",
        multiplier,
        False,
        0,
        0.0,
        3,
    ]

    return arg_list

def get_tiled_vae_args():
    arg_list =  [
        True,
        2048,
        128,
        True,
        True,
        True,
        True,
    ]
    return arg_list

def get_controlnet_args(base64_image, resemblance):
    arg_dict = {
        "enabled": True,
        "module": "tile_resample",
        "model": "control_v11f1e_sd15_tile",
        "weight": resemblance,
        "image": base64_image,
        "resize_mode": 1,
        "lowvram": False,
        "downsample": 1.0,
        "guidance_start": 0.0,
        "guidance_end": 1.0,
        "control_mode": 1,
        "pixel_perfect": True,
        "threshold_a": 1,
        "threshold_b": 1,
        "save_detected_map": False,
        "processor_res": 512,
    }
    arg_list = [arg_dict]

    return arg_list
