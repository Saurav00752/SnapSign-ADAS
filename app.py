import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# Simple dictionary for common classes (expand as needed)
classes = {0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
           14: 'Stop', 25: 'Road work', 31: 'Wild animals crossing'}

st.title("🚗 Autonomous Perception: Traffic Sign CNN")
st.write("Upload a dashcam frame to classify the traffic sign in real-time.")

@st.cache_resource # Caches the model so it doesn't reload on every interaction
def load_model():
    model = get_model(num_classes=43)
    # map_location='cpu' ensures it runs even if the judge doesn't have a GPU
    model.load_state_dict(torch.load('traffic_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Frame', use_column_width=True)
    
    st.write("Processing...")
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
        class_id = predicted.item()
        
    sign_name = classes.get(class_id, f"Class ID {class_id}")
    st.success(f"Prediction: **{sign_name}**")