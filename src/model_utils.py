import os
import zipfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "snapsign_optimized.onnx")
MODEL_ARCHIVE = os.path.join(MODELS_DIR, "snapsign_optimized.onnx.onnx.zip")


def ensure_optimized_model():
    """Return the local ONNX path, extracting the downloaded AI Hub bundle if needed."""
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    if not os.path.isfile(MODEL_ARCHIVE):
        return None

    os.makedirs(MODELS_DIR, exist_ok=True)
    with zipfile.ZipFile(MODEL_ARCHIVE) as archive:
        members = {
            os.path.basename(name): name
            for name in archive.namelist()
            if os.path.basename(name) in {"model.onnx", "model.data"}
        }
        if "model.onnx" not in members:
            raise FileNotFoundError("The optimized model archive does not contain model.onnx")
        for filename, member in members.items():
            destination = os.path.join(MODELS_DIR, filename)
            with archive.open(member) as source, open(destination, "wb") as target:
                target.write(source.read())
        os.replace(os.path.join(MODELS_DIR, "model.onnx"), MODEL_PATH)
    return MODEL_PATH
