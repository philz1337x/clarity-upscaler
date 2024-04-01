from modules import timer
from modules import launch_utils
from modules import initialize_util
from modules import initialize
from fastapi import FastAPI
import base64
import os, sys, json
from PIL import Image
import uuid
from io import BytesIO

import cv2
import numpy as np
from urllib.parse import urlparse
import requests

import time

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ['IGNORE_CMD_ARGS_ERRORS'] = '1'

        startup_timer = timer.startup_timer
        startup_timer.record("launcher")
        
        initialize.imports()
        initialize.check_versions()
        initialize.initialize()
        
        app = FastAPI()
        initialize_util.setup_middleware(app)
        
        from modules.api.api import Api
        from modules.call_queue import queue_lock
        
        self.api = Api(app, queue_lock)
        
        from modules import script_callbacks
        script_callbacks.before_ui_callback()
        script_callbacks.app_started_callback(None, app)
        
        print(f"Startup time: {startup_timer.summary()}.")

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

        safetensors_path = "models/Stable-diffusion/custom.safetensors"

        response = requests.get(url)
        response.raise_for_status()

        with open(safetensors_path, "wb") as file:
            file.write(response.content)

        print(f"Custom checkpoint downloading and saving took {time.time() - start_time_custom} seconds")

        return safetensors_path

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
            choices=['epicrealism_naturalSinRC1VAE.safetensors [84d76a0328]', 'juggernaut_reborn.safetensors [338b85bc4f]', 'juggernaut_final.safetensors', 'flat2DAnimerge_v45Sharp.safetensors'],
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
            description="Link to a custom safetensors checkpoint file you want to use in your upscaling. Will overwrite sd_model checkpoint.",
            default=""
        ),
    ) -> list[Path]:
        """Run a single prediction on the model"""
        print("Running prediction")
        if lora_links:
            lora_link = [link.strip() for link in lora_links.split(",")]
            for link in lora_link:
                self.download_lora_weights(link) 

        if custom_sd_model:
            path_to_custom_checkpoint = self.download_safetensors(custom_sd_model)
            sd_model = "custom.safetensors"
            self.api.refresh_checkpoints()

        from modules.api.models import StableDiffusionImg2ImgProcessingAPI

        image_file_path = image
           
        with open(image_file_path, "rb") as image_file:
            binary_image_data = image_file.read()

        if downscaling:
            image_np_array = np.frombuffer(binary_image_data, dtype=np.uint8)

            image = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)

            height, width = image.shape[:2]
            
            if height > width:
                scaling_factor = downscaling_resolution / float(height)
            else:
                scaling_factor = downscaling_resolution / float(width)
            
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)

            resized_image = cv2.resize(image, (new_width, new_height))

            _, binary_resized_image = cv2.imencode('.jpg', resized_image)
            binary_image_data = binary_resized_image.tobytes()

        base64_encoded_data = base64.b64encode(binary_image_data)
        base64_image = base64_encoded_data.decode('utf-8')

        payload = {
            "override_settings": {
                "sd_model_checkpoint": sd_model,
                "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
                 "CLIP_stop_at_last_layers": 1,
            },
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
            "alwayson_scripts": {
                "Tiled Diffusion": {
                    "args": [
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
                        scale_factor, 
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
                    ]
                }
            }
        }

        req = StableDiffusionImg2ImgProcessingAPI(**payload)

        resp = self.api.img2imgapi(req)
        info = json.loads(resp.info)

        outputs = []

        for i, image in enumerate(resp.images):
            seed = info.get("all_seeds", [])[i] or "unknown_seed"
            gen_bytes = BytesIO(base64.b64decode(image))
            filename = f"{seed}-{uuid.uuid1()}.png"
            with open(filename, "wb") as f:
                f.write(gen_bytes.getvalue())
            outputs.append(Path(filename))
        
        if custom_sd_model:
            os.remove(path_to_custom_checkpoint)
            print(f"Custom checkpoint {path_to_custom_checkpoint} has been removed.")

        return outputs
    