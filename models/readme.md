# SnapSign Models

This directory is intended to store the locally trained and optimized model weights. 

To keep the repository lightweight and comply with GitHub's file size limits, the raw `.pth` and `.onnx` files are excluded via `.gitignore`. 

## How to generate the models:
1. Run `src/train.py` to train the base model on GTSRB and generate `snapsign_base.pth`.
2. Run `src/compile_qai.py` to pass the base model through Qualcomm AI Hub and generate the NPU-optimized `snapsign_optimized.onnx`.