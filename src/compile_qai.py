import torch
import qai_hub
import os
from train import SnapSignNet

def compile_for_snapdragon():
    # 1. Load the trained PyTorch model
    model = SnapSignNet(num_classes=43)
    model.load_state_dict(torch.load("../models/snapsign_base.pth"))
    model.eval()

    # 2. Trace the model (required by AI Hub)
    dummy_input = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model, dummy_input)

    print("Submitting model to Qualcomm AI Hub for Snapdragon X Elite compilation...")
    
    # 3. Submit compile job for Snapdragon HP PCs
    compile_job = qai_hub.submit_compile_job(
        model=traced_model,
        device=qai_hub.Device("Snapdragon X Elite Compute Platform"),
        input_specs=dict(image=(1, 3, 32, 32)),
        options="--target_runtime onnx"
    )

    # 4. Download and save the optimized ONNX model
    optimized_model = compile_job.get_target_model()
    
    os.makedirs("../models", exist_ok=True)
    with open("../models/snapsign_optimized.onnx", "wb") as f:
        f.write(optimized_model)
        
    print("Success! Optimized model saved to models/snapsign_optimized.onnx")

if __name__ == "__main__":
    compile_for_snapdragon()