import os
import requests

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