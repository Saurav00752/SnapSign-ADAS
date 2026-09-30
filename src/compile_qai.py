import torch
import qai_hub
import os
from train import SnapSignNet
from model_utils import MODELS_DIR

def compile_for_snapdragon():
    # 1. Load the trained PyTorch model
    model = SnapSignNet(num_classes=43)
    base_model_path = os.path.join(MODELS_DIR, "snapsign_base.pth")
    model.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    model.eval()

    # 2. Export the model using the modern PyTorch 2.x method
    dummy_input = torch.randn(1, 3, 32, 32)
    exported_model = torch.export.export(model, (dummy_input,))

    print("Submitting model to Qualcomm AI Hub for Snapdragon X Elite compilation...")
    
    # 3. Submit compile job using the correct exact CRD device string
    compile_job = qai_hub.submit_compile_job(
        model=exported_model,
        device=qai_hub.Device("Snapdragon X Elite CRD"),
        options="--target_runtime onnx"
    )

    # 4. Download and save the optimized ONNX model using the built-in method
    optimized_model = compile_job.get_target_model()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    optimized_model.download(os.path.join(MODELS_DIR, "snapsign_optimized.onnx"))
        
    print("Success! Optimized model saved to models/snapsign_optimized.onnx")

if __name__ == "__main__":
    compile_for_snapdragon()