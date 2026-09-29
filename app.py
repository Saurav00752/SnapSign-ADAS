import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import pandas as pd
from model import get_model

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="Snapdragon Edge AI Perception", 
    page_icon="🏎️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Complete GTSRB Dictionary
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing for heavy vehicles', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
    16: 'Heavy vehicles prohibited', 17: 'No entry', 18: 'General caution', 
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 
    41: 'End of no passing', 42: 'End of no passing by heavy vehicles'
}

# 3. Cache Model Loading
@st.cache_resource
def load_model():
    model = get_model(num_classes=43)
    model.load_state_dict(torch.load('traffic_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Dashboard Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Qualcomm-Logo.svg/1200px-Qualcomm-Logo.svg.png", width=150)
st.sidebar.title("Edge AI Settings")
hardware_target = st.sidebar.radio(
    "Simulated Hardware Target",
    ["Snapdragon® 8 Gen 3 NPU", "Hexagon™ DSP", "CPU Fallback"]
)
st.sidebar.markdown("---")
st.sidebar.write("**Model Architecture:** Fine-tuned ResNet18")
st.sidebar.write("**Precision:** FP32 (Ready for INT8 Quantization)")
st.sidebar.write("**Input Resolution:** 224x224 RGB")

# 5. Main Dashboard Header
st.title("🏎️ Autonomous Perception Module")
st.markdown(f"Running on: **{hardware_target}** | Optimizing for sub-50ms latency.")

# 6. Two-Column Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📷 Sensor Input")
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Dashcam Frame", "Live Edge Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Processed Sensor Data', use_container_width=True)
            
    with tab2:
        camera_file = st.camera_input("Take a picture")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')
            uploaded_file = camera_file # Route it to the same logic

with col2:
    st.subheader("🧠 Telemetry & Inference")
    if uploaded_file is not None:
        # Preprocessing
        input_tensor = transform(image).unsqueeze(0) 
        
        # Inference with Latency Tracking
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
        end_time = time.perf_counter()
        
        # Calculate Metrics
        latency_ms = (end_time - start_time) * 1000
        fps = 1000 / latency_ms if latency_ms > 0 else 0
        
        # Get Top 3 Predictions
        top_prob, top_class = probabilities.topk(3, dim=1)
        top_prob = top_prob.squeeze().numpy() * 100
        top_class = top_class.squeeze().numpy()
        
        primary_prediction = classes.get(top_class[0], "Unknown")
        primary_confidence = top_prob[0]

        # Display Hero Metric
        st.metric(label="Primary Classification", value=primary_prediction, delta=f"{primary_confidence:.1f}% Confidence")
        
        # Display Hardware Telemetry
        st.markdown("### ⚡ Edge Performance Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Latency", f"{latency_ms:.1f} ms", delta="- Optimized" if latency_ms < 100 else "High")
        mcol2.metric("Est. FPS", f"{fps:.1f}", delta="+ Real-time" if fps > 24 else None)
        mcol3.metric("Memory Footprint", "44.6 MB")
        
        # Display Confidence Chart
        st.markdown("### 📊 Classification Probability (Top 3)")
        chart_data = pd.DataFrame({
            "Confidence (%)": top_prob
        }, index=[classes.get(c, f"Class {c}") for c in top_class])
        
        st.bar_chart(chart_data)
        
    else:
        st.info("Awaiting sensor input... Upload an image or activate the camera on the left.")