import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class VeloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = img_name.replace('.jpg', '.txt')
        
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        
        # Таргет 7x7 с 5 параметрами: [conf, x, y, w, h]
        target = torch.zeros((7, 7, 5))
        
        label_path = os.path.join(self.label_dir, label_name)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO: class x_c y_c w h
                    _, x, y, w, h = map(float, line.split())
                    
                    # Определяем ячейку сетки
                    grid_x = int(x * 7)
                    grid_y = int(y * 7)
                    
                    # Локальные координаты внутри ячейки
                    x_local = x * 7 - grid_x
                    y_local = y * 7 - grid_y
                    
                    target[grid_y, grid_x, 0] = 1.0 # Conf
                    target[grid_y, grid_x, 1:] = torch.tensor([x_local, y_local, w, h])

        if self.transform:
            image = self.transform(image)
            
        return image, target