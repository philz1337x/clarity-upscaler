import os
import requests
import shutil

def download_file(url, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        print(f"File already exists: {file_path}")
    else:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"File successfully downloaded and saved: {file_path}")
        else:
            print(f"Error downloading the file. Status code: {response.status_code}")

# Prepare webui
from modules.launch_utils import prepare_environment
prepare_environment()

print("Modifiying controlnet.py")
shutil.copyfile('modified_controlnet.py', 'extensions/sd-webui-controlnet/scripts/controlnet.py')
print("Modifiying controlnet.py - Done")

# Checkpoints
download_file(
    "https://huggingface.co/philz1337x/flat2DAnimerge_v45Sharp/resolve/main/flat2DAnimerge_v45Sharp.safetensors?download=true",
    "models/Stable-diffusion",
    "flat2DAnimerge_v45Sharp.safetensors"
)
download_file(
    "https://huggingface.co/dantea1118/juggernaut_reborn/resolve/main/juggernaut_reborn.safetensors?download=true",
    "models/Stable-diffusion",
    "juggernaut_reborn.safetensors"
)
download_file(
    "https://huggingface.co/philz1337x/epicrealism/resolve/main/epicrealism_naturalSinRC1VAE.safetensors?download=true",
    "models/Stable-diffusion",
    "epicrealism_naturalSinRC1VAE.safetensors"
)

# Upscaler Model
download_file(
    "https://huggingface.co/philz1337x/upscaler/resolve/main/4x-UltraSharp.pth?download=true",
    "models/ESRGAN",
    "4x-UltraSharp.pth"
)

# Embeddings
download_file(
    "https://huggingface.co/philz1337x/embeddings/resolve/main/verybadimagenegative_v1.3.pt?download=true",
    "embeddings",
    "verybadimagenegative_v1.3.pt"
)
download_file(
    "https://huggingface.co/philz1337x/embeddings/resolve/main/JuggernautNegative-neg.pt?download=true",
    "embeddings",
    "JuggernautNegative-neg.pt"
)

# Lora Models
download_file(
    "https://huggingface.co/philz1337x/loras/resolve/main/SDXLrender_v2.0.safetensors?download=true",
    "models/Lora",
    "SDXLrender_v2.0.safetensors"
)
download_file(
    "https://huggingface.co/philz1337x/loras/resolve/main/more_details.safetensors?download=true",
    "models/Lora",
    "more_details.safetensors"
)

# Controlnet models
download_file(
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth?download=true",
    "models/ControlNet",
    "control_v11f1e_sd15_tile.pth"
)

# VAE
download_file(
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true",
    "models/VAE",
    "vae-ft-mse-840000-ema-pruned.safetensors"
)
