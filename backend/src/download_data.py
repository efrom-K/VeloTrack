import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, save_path):
    # allow_redirects=True критично для GitHub/S3 ссылок
    response = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if total_size < 1000:
        print(f"Warning: File size is too small ({total_size} bytes). Check URL.")
        
    with open(save_path, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc="Downloading"
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def prepare_data():
    data_dir = "./data/cars"
    os.makedirs(data_dir, exist_ok=True)
    
    # Используем прямую ссылку на датасет (Car Detection для YOLO)
    url = "https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/car-detection.zip"
    zip_path = os.path.join(data_dir, "cars.zip")
    
    # Принудительно удалим старый битый файл, если он есть
    if os.path.exists(zip_path) and os.path.getsize(zip_path) < 1000:
        os.remove(zip_path)

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        download_file(url, zip_path)
        
    print(f"Unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Done! Data is ready in ./data/cars")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is corrupt. Try deleting ./data/cars/cars.zip and run again.")

if __name__ == "__main__":
    prepare_data()