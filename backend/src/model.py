import torch
import torch.nn as nn
import torchvision
from torchvision import models

class VeloTrackDetector(nn.Module):
    def __init__(self):
        super(VeloTrackDetector, self).__init__()
        # Используем ResNet18 как экстрактор признаков
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Убираем последние два слоя (avgpool и fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Голова детектора
        # Вход: 512 каналов из ResNet. Выход: сетка 7x7 по 5 параметров (conf, x, y, w, h)
        self.detector = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1) # 5 каналов: [P, x, y, w, h]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x) # На выходе [B, 512, 7, 7]
        x = self.detector(x) # На выходе [B, 5, 7, 7]
        x = self.sigmoid(x)
        return x.permute(0, 2, 3, 1) # Перекладываем в [B, 7, 7, 5] для удобства