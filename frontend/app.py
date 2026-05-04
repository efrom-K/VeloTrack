import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision
import os
from model import VeloTrackDetector

# Настройки страницы
st.set_page_config(page_title="VeloTrack AI", layout="wide")

@st.cache_resource
def load_model():
    # Определяем устройство (твоя RTX 3060 или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VeloTrackDetector()
    
    # Путь к весам в корневой папке
    weights_path = "/app/weights/model_weights.pth"
    
    if os.path.exists(weights_path):
        # Загружаем веса
        model.load_state_dict(torch.load(weights_path, map_location=device))
        st.sidebar.success("Обученные веса загружены")
    else:
        st.sidebar.warning("Файл весов не найден. Используются случайные значения.")
        
    model.to(device).eval()
    return model, device

def process_image(image, model, device, threshold):
    # Подготовка изображения
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor) # [1, 7, 7, 5]

    boxes = []
    scores = []
    
    h_img, w_img, _ = np.array(image).shape
    grid_size = 7
    
    # Сбор детекций выше порога
    for i in range(grid_size):
        for j in range(grid_size):
            conf, dx, dy, dw, dh = output[0, i, j].cpu().numpy()
            if conf > threshold:
                # Масштабируем относительные координаты в пиксели оригинала
                cx = ((j + dx) / grid_size) * w_img
                cy = ((i + dy) / grid_size) * h_img
                bw = dw * w_img
                bh = dh * h_img
                
                # Формат [x1, y1, x2, y2]
                boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
                scores.append(float(conf))

    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    if len(boxes) > 0:
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        
        # NMS фильтрация
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.4)
        
        final_boxes = boxes_tensor[keep].numpy()
        final_scores = scores_tensor[keep].numpy()

        # Отрисовка отфильтрованных боксов
        for box, score in zip(final_boxes, final_scores):
            x1, y1, x2, y2 = box.astype(int)
            
            # Ограничиваем координаты границами изображения
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"Car: {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        count = len(keep)
    else:
        count = 0

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), count

# Интерфейс
st.title("🚗 VeloTrack: Car Counter")

# Загружаем модель один раз
model, device = load_model()

with st.sidebar:
    st.header("Настройки")
    threshold = st.slider("Порог уверенности", 0.0, 1.0, 0.45)
    st.info(f"Запущено на: {device}")

uploaded_file = st.file_uploader("Загрузите фото дороги...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Оригинал", use_container_width=True)
    
    # Кнопка запуска
    if st.button("Запустить анализ", use_container_width=True):
        with st.spinner("Нейросеть думает..."):
            result_img, car_count = process_image(image, model, device, threshold)
            
            with col2:
                st.image(result_img, caption="Результат", use_container_width=True)
                st.metric("Найдено автомобилей", car_count)
                
                if car_count == 0:
                    st.info("Попробуйте снизить порог уверенности.")