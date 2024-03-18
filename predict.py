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
            description="Resemblance, try from 0.3 - 1.6", default=0.6
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
        )
    ) -> list[Path]:
        """Run a single prediction on the model"""

        from modules.api.models import StableDiffusionImg2ImgProcessingAPI
        
        with open(image, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode()

        payload = {
            "override_settings": {
                "sd_model_checkpoint": "juggernaut_reborn.safetensors [338b85bc4f]",
                "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
                 "CLIP_stop_at_last_layers": 1,
            },
            "override_settings_restore_afterwards": False,
            "init_images": [img_base64],
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
                            "image": img_base64,
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
            
        return outputs
    