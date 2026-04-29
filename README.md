# VeloTrack: Car Detection & Counting System

A lightweight, high-performance computer vision system for real-time vehicle detection and counting. Built with a focus on modularity and scalable deployment using Docker and NVIDIA GPU acceleration.

## 🚀 Overview
VeloTrack uses a custom Object Detection architecture based on a **ResNet18 backbone**. The system processes images through a $7 \times 7$ grid where each cell predicts object confidence and bounding box coordinates $[P, x, y, w, h]$.

### Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Frontend:** Streamlit (Python-based UI)
* **Backend:** Python 3.10 + CUDA 12.1
* **Image Processing:** OpenCV
* **Infrastructure:** Docker, Docker Compose
* **Hardware Acceleration:** NVIDIA GPU (RTX 3060 tested) via NVIDIA Container Toolkit

## 🛠 Project Structure
```text
.
├── frontend/
│   ├── app.py          # Streamlit UI & Inference logic
│   ├── model.py        # Shared architecture definition
│   └── Dockerfile      # Frontend container config
├── backend/
│   ├── src/
│   │   ├── train.py    # Training loop & logic
│   │   └── model.py    # Detector architecture
│   └── Dockerfile      # CUDA-enabled backend container
└── docker-compose.yml  # Orchestration & GPU passthrough
```
## ⚡ Quick Start
Prerequisites
* Docker & Docker Compose
* NVIDIA Container Toolkit (for GPU support)

## Deployment
Clone the repository:

```Bash
git clone [https://github.com/yourusername/VeloTrack.git](https://github.com/yourusername/VeloTrack.git)
cd VeloTrack
```
Launch the stack:

```Bash
docker-compose up --build
```
Access the Web UI:
Open http://localhost:8501 in your browser.

## 🧠 Model Details
The current implementation uses a Grid-based Detector strategy:
* Backbone: ResNet18 (Pretrained on ImageNet)
* Head: Custom Conv2D layers outputting a [5, 7, 7] tensor.
* Coordinate System: Relative offsets (dx, dy) from grid cell origin and normalized width/height (dw, dh).

## 📈 Roadmap
[x] Dockerized Infrastructure & GPU passthrough.

[x] Grid-based Detection UI implementation.

[ ] Implement Non-Maximum Suppression (NMS) for box filtering.

[ ] Integrate COCO/Stanford Cars dataset for training.

[ ] Transition to YOLO-style Loss Function (Localization + Confidence).
---
*Developed as part of an experimental AI vision project.*