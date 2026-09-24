# 🚗 Autonomous Perception: Real-Time Traffic Sign Classification

An AI-powered perception module designed for autonomous vehicles. This project uses a fine-tuned ResNet Convolutional Neural Network (CNN) to classify German Traffic Signs (GTSRB) with high accuracy, wrapped in a real-time Streamlit web interface.

## ⚙️ Features
* **Transfer Learning:** Utilizes ResNet18 backbone fine-tuned for 43 traffic sign classes.
* **Hardware-Agnostic Inference:** Model defaults to CPU inference in the app, ensuring it runs on any judge's machine.
* **Interactive UI:** Upload any dashcam frame or street image for instant classification.

## 🚀 How to Run Locally

**1. Clone the repository**

**2. Create a virtual environment and install dependencies**


**3. Run the Web App**

*Note: The repository includes pre-trained weights (`traffic_weights.pth`). If you want to train the model from scratch, run `python train.py`.*

## 🗂️ Repository Structure
* `model.py` - CNN architecture (ResNet18 modifications)
* `train.py` - Data ingestion, augmentation, and training loop
* `app.py` - Streamlit web application
* `requirements.txt` - Python dependencies