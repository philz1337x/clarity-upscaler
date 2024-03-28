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

# Download the first file
url_1 = "https://huggingface.co/philz1337x/upscaler/resolve/main/RealESRGAN_x4plus_anime_6B.pth?download=true"
folder_path_1 = "models/ESRGAN"
filename_1 = "RealESRGAN_x4plus_anime_6B.pth"
download_file(url_1, folder_path_1, filename_1)

# Download the second file
url_2 = "https://huggingface.co/philz1337x/embeddings/resolve/main/verybadimagenegative_v1.3.pt?download=true"
folder_path_2 = "embeddings"
filename_2 = "verybadimagenegative_v1.3.pt"
download_file(url_2, folder_path_2, filename_2)
