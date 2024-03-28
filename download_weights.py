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

#upscaler
url_1 = "https://huggingface.co/philz1337x/upscaler/resolve/main/4x-UltraSharp.pth?download=true"
folder_upscaler = "models/ESRGAN"
filename_1 = "RealESRGAN_x4plus_anime_6B.pth"
download_file(url_1, folder_upscaler, filename_1)

#embeddings
url_2 = "https://huggingface.co/philz1337x/embeddings/resolve/main/verybadimagenegative_v1.3.pt?download=true"
folder_embeddings = "embeddings"
filename_2 = "verybadimagenegative_v1.3.pt"
download_file(url_2, folder_embeddings, filename_2)

#lora
url_3 = "https://huggingface.co/philz1337x/loras/resolve/main/SDXLrender_v2.0.safetensors?download=true"
folder_lora = "models/Lora"
filename_3 = "SDXLrender_v2.0.safetensors"
download_file(url_3, folder_lora, filename_3)

url_4 = "https://huggingface.co/philz1337x/loras/resolve/main/more_details.safetensors?download=true"
filename_4 = "more_details.safetensors"
download_file(url_4, folder_lora, filename_4)