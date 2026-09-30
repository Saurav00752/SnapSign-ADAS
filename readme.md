# 🚗 SnapSign ADAS: Edge-Optimized Perception for Snapdragon PCs

**Snapdragon AI Lab Hackathon Submission**

SnapSign ADAS is a real-time Advanced Driver Assistance System (ADAS) perception module designed exclusively for **Snapdragon-powered HP PCs**. It processes dashcam video feeds locally to classify traffic signs with zero cloud latency, absolute data privacy, and minimal power consumption. 

By leveraging the **Qualcomm AI Hub** and the **Hexagon NPU** on the Snapdragon X Elite/Plus architecture, this project demonstrates how self-driving perception pipelines can be efficiently ported to Windows on ARM.

---

## 🌟 Hackathon Evaluation Criteria Mapping

This project was built to address the core evaluation criteria of the Snapdragon AI Edge Hackathon:

1. **Application Use Case & Innovation:** Brings critical automotive safety features (traffic sign recognition) to the edge. Functions entirely offline, simulating a localized vehicle NPU environment on an HP Omnibook.
2. **Technical Implementation:** Transitions a standard PyTorch CNN into an optimized Edge AI module. Utilizes `qai-hub` for model compilation and INT8 quantization, targeting the specific Snapdragon X hardware.
3. **Deployment & Accessibility:** Runs natively on Windows ARM64 using ONNX Runtime and the QNN Execution Provider, ensuring maximum FPS without x86 emulation overhead.
4. **Presentation & Documentation:** Clean, modular repository architecture with reproducible scripts from training to edge inference.

---

## 🏗️ System Architecture

The pipeline follows a strict edge-deployment lifecycle:

1. **Train:** A lightweight CNN trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset using PyTorch.
2. **Compile & Quantize:** The `.pth` model is traced and pushed through the **Qualcomm AI Hub API**, which quantizes and compiles the model specifically for the Snapdragon X Elite Compute Platform.
3. **Infer:** Real-time video frames are processed using `onnxruntime` routed through the **QNNExecutionProvider**, offloading the heavy matrix math directly to the laptop's Hexagon NPU.

---

## 📂 Repository Structure

```text
SnapSign-ADAS/
├── models/
│   ├── snapsign_base.pth        # Original PyTorch weights
│   └── snapsign_optimized.onnx  # AI Hub compiled & quantized model
├── src/
│   ├── train.py                 # GTSRB PyTorch training script
│   ├── compile_qai.py           # Qualcomm AI Hub submission script
│   └── infer_qnn.py             # Real-time ONNX/QNN NPU inference script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes datasets and virtual environments
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
* A Snapdragon-powered PC (e.g., HP Omnibook Ultra/X with Snapdragon X Elite/Plus)
* Windows 11 on ARM
* Python 3.10+ (Native ARM64 version)
* Qualcomm AI Hub API Token

### 1. Install Dependencies
Clone the repository and install the required packages:
```bash
git clone https://github.com/yourusername/SnapSign-ADAS.git
cd SnapSign-ADAS
pip install -r requirements.txt
```
*(Ensure you have `qai-hub` and `onnxruntime` installed natively for ARM64).*

### 2. Compile for Snapdragon NPU
If you wish to re-compile the model for your specific Snapdragon chip, configure your Qualcomm AI Hub token and run:
```bash
qai-hub configure --api_token YOUR_API_TOKEN
python src/compile_qai.py
```
This will output `snapsign_optimized.onnx` into the `/models` directory.

### 3. Run Real-Time Inference
Run the ADAS simulation on a sample dashcam video or live webcam feed:
```bash
python src/infer_qnn.py --video data/sample_dashcam.mp4
```
*Note: The script automatically detects the Hexagon NPU via the `QNNExecutionProvider`.*

---

## 📊 Performance & Benchmarks (Expected)

By utilizing the Qualcomm AI Hub and NPU offloading, SnapSign ADAS achieves:
* **Higher FPS:** Significant inference speedup compared to running the raw PyTorch model on the CPU.
* **Lower Power Draw:** Preserves laptop battery life, demonstrating the feasibility of always-on ADAS monitoring on edge devices.

---

## 🤝 Acknowledgments
* **Qualcomm AI Hub:** For seamless edge model compilation.
* **GTSRB Dataset:** For the robust traffic sign training data.
* **Snapdragon AI Lab Hackathon:** For inspiring this edge-AI innovation.

*Developed for the Snapdragon PC Ecosystem.*