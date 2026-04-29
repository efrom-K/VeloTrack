import torch
import torch.optim as optim
import torch.nn as nn
# Было: from model import VeloTrackNet
from model import VeloTrackDetector 

def train():
    # Инициализируем нашу новую модель
    model = VeloTrackDetector().cuda()
    
    # Для детекции нам нужна специфическая функция потерь (MSE для координат + BCE для наличия объекта)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() 

    print("Модель готова к обучению. Ожидание данных...")
    
    # Пока здесь заглушка, чтобы контейнер не вылетал
    try:
        while True:
            pass 
    except KeyboardInterrupt:
        print("Остановка...")

if __name__ == "__main__":
    train()