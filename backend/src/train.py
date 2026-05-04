import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import VeloTrackDetector
from dataset import VeloDataset

# 1. Лосс-функция для детекции (YOLO-style)
class VeloLoss(nn.Module):
    def __init__(self):
        super(VeloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        # targets/predictions: [batch, 7, 7, 5] -> [conf, x, y, w, h]
        exists_filter = targets[..., 0] == 1
        no_exists_filter = targets[..., 0] == 0

        # Loss для ячеек с объектами
        if exists_filter.any():
            p_box = predictions[exists_filter][:, 1:]
            t_box = targets[exists_filter][:, 1:]
            loss_coord = self.mse(p_box, t_box)

            p_conf = predictions[exists_filter][:, 0]
            t_conf = targets[exists_filter][:, 0]
            loss_conf = self.bce(p_conf, t_conf)
        else:
            loss_coord = 0
            loss_conf = 0

        # Loss для пустых ячеек (штраф за галлюцинации)
        p_noobj = predictions[no_exists_filter][:, 0]
        t_noobj = targets[no_exists_filter][:, 0]
        loss_noobj = self.bce(p_noobj, t_noobj)

        return self.lambda_coord * loss_coord + loss_conf + self.lambda_noobj * loss_noobj

# 2. Основной цикл обучения
def train():
    # Настройки устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Гиперпараметры
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DATA_DIR = "./data/cars"

    # Трансформации (обязательно 224x224 как в ResNet18)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Инициализация данных
    train_dataset = VeloDataset(
        img_dir=f"{DATA_DIR}/train/images",
        label_dir=f"{DATA_DIR}/train/labels",
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Инициализация модели
    model = VeloTrackDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = VeloLoss()

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Сохраняем промежуточный чекпоинт каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"velo_model_e{epoch+1}.pth")

    # Финальное сохранение
    torch.save(model.state_dict(), "model_weights.pth")
    print("Training finished. Weights saved to model_weights.pth")

if __name__ == "__main__":
    train()