
from modules import timer
from modules import initialize_util
from modules import initialize
from urllib.parse import urlparse
from fastapi import FastAPI
from io import BytesIO

import os, json
import numpy as np
import requests
import base64
import uuid
import time
import cv2

from cog import BasePredictor, Input, Path

from PIL import Image, ImageFilter
import tempfile

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
        
        model_response = self.api.get_sd_models()
        print("Available checkpoints: ", str(model_response))

        from modules import script_callbacks
        script_callbacks.before_ui_callback()
        script_callbacks.app_started_callback(None, app)

        from modules.api.models import StableDiffusionImg2ImgProcessingAPI
        self.StableDiffusionImg2ImgProcessingAPI = StableDiffusionImg2ImgProcessingAPI

        file_path = Path("init.png")
        base64_encoded_data = base64.b64encode(file_path.read_bytes())
        base64_image = base64_encoded_data.decode('utf-8')

        
        
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
        
        req = StableDiffusionImg2ImgProcessingAPI(**payload)
        self.api.img2imgapi(req)

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
        safetensors_path = f"models/Stable-diffusion/custom-{uuid.uuid1()}.safetensors"

        response = requests.get(url)
        response.raise_for_status()

        with open(safetensors_path, "wb") as file:
            file.write(response.content)

        print(f"Checkpoint downloading took {round(time.time() - start_time_custom, 2)} seconds")

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
    ) -> list[Path]:
        """Run a single prediction on the model"""
        print("Running prediction")
        start_time = time.time()
        
        # checkpoint name changed bc hashing is deactivated so name is corrected here to old name to avoid breaking api calls
        if sd_model == "epicrealism_naturalSinRC1VAE.safetensors [84d76a0328]":
            sd_model = "epicrealism_naturalSinRC1VAE.safetensors"
        if sd_model == "juggernaut_reborn.safetensors [338b85bc4f]":
            sd_model = "juggernaut_reborn.safetensors"
    
        if lora_links:
            lora_link = [link.strip() for link in lora_links.split(",")]
            for link in lora_link:
                self.download_lora_weights(link) 

        if custom_sd_model:
            path_to_custom_checkpoint = self.download_safetensors(custom_sd_model)
            sd_model = os.path.basename(path_to_custom_checkpoint)
            self.api.refresh_checkpoints()
        
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

        multipliers = [scale_factor]
        if scale_factor > 2:
            multipliers = self.calc_scale_factors(scale_factor)
            print("Upscale your image " + str(len(multipliers)) + " times")
        
        first_iteration = True

        for multiplier in multipliers:
            print("Upscaling with scale_factor: ", multiplier)
            
            if not first_iteration:
                creativity = creativity * 0.8
                seed = seed +1
                
            first_iteration = False

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
                            multiplier, 
                            False, 
                            0,
                            0.0, 
                            3,
                        ]
                    },
                    "Tiled VAE": {
                        "args": [
                            True,
                            2048,
                            128,
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

            req = self.StableDiffusionImg2ImgProcessingAPI(**payload)
            resp = self.api.img2imgapi(req)
            info = json.loads(resp.info)

            base64_image = resp.images[0]

            outputs = []

            for i, image in enumerate(resp.images):
                seed = info.get("all_seeds", [])[i] or "unknown_seed"
                gen_bytes = BytesIO(base64.b64decode(image))
                filename = f"{seed}-{uuid.uuid1()}.png"
                with open(filename, "wb") as f:
                    f.write(gen_bytes.getvalue())

                if sharpen > 0:
                    imageObject = Image.open(filename)

                    a = -sharpen / 10
                    b = 1 - 8 * a
                    kernel = [a, a, a, a, b, a, a, a, a]
                    kernel_filter = ImageFilter.Kernel((3, 3), kernel, scale=1, offset=0)

                    imageObject = imageObject.filter(kernel_filter)

                    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    imageObject.save(temp_file_path, format='PNG')
                    temp_file_path.close()
                    
                    outputs.append(Path(temp_file_path.name))
                else:
                    outputs.append(Path(filename))
        
        if custom_sd_model:
            os.remove(path_to_custom_checkpoint)
            print(f"Custom checkpoint {path_to_custom_checkpoint} has been removed.")

        print(f"Prediction took {round(time.time() - start_time,2)} seconds")
        return outputs
    