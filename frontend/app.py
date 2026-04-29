import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import VeloTrackDetector

# Настройки страницы
st.set_page_config(page_title="VeloTrack AI", layout="wide")

@st.cache_resource
def load_model():
    model = VeloTrackDetector()
    # Если есть сохраненные веса, раскомментируй:
    # model.load_state_dict(torch.load("/app/weights/best_model.pth", map_location='cuda'))
    model.cuda().eval()
    return model

def process_image(image, model, threshold):
    # Подготовка изображения
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transforms(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        # [1, 7, 7, 5] -> (conf, x, y, w, h)
        output = model(img_tensor)

    # Отрисовка
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_cv.shape
    
    count = 0
    grid_size = 7
    
    for i in range(grid_size):
        for j in range(grid_size):
            conf, dx, dy, dw, dh = output[0, i, j].cpu().numpy()
            
            if conf > threshold:
                count += 1
                # Вычисляем координаты центра в пикселях
                # j - столбец (x), i - строка (y)
                center_x = int(((j + dx) / grid_size) * w_img)
                center_y = int(((i + dy) / grid_size) * h_img)
                
                # Ширина и высота бокса
                box_w = int(dw * w_img)
                box_h = int(dh * h_img)
                
                # Координаты углов
                x1, y1 = center_x - box_w // 2, center_y - box_h // 2
                x2, y2 = center_x + box_w // 2, center_y + box_h // 2
                
                # Рисуем рамку и текст
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv, f"{conf:.2f}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), count

# Интерфейс
st.title("🚗 VeloTrack: Car Counter")

with st.sidebar:
    st.header("Настройки")
    threshold = st.slider("Порог уверенности", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Загрузите фото дороги...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Оригинал", use_container_width=True)
    
    if st.button("Запустить анализ"):
        model = load_model()
        result_img, car_count = process_image(image, model, threshold)
        
        with col2:
            st.image(result_img, caption="Результат", use_container_width=True)
            st.metric("Найдено автомобилей", car_count)
            if car_count == 0:
                st.info("Попробуйте снизить порог уверенности (модель пока не обучена)")