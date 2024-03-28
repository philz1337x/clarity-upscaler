import os
import requests

def download_file(url, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File successfully downloaded and saved: {file_path}")
    else:
        print(f"Error downloading the file. Status code: {response.status_code}")

url = "https://huggingface.co/philz1337x/upscaler/resolve/main/RealESRGAN_x4plus_anime_6B.pth?download=true"
folder_path = "models/ESRGAN"
filename = "RealESRGAN_x4plus_anime_6B.pth"
download_file(url, folder_path, filename)
