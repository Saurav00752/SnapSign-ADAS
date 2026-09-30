import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import os
from model_utils import ensure_optimized_model

def run_npu_inference(video_source=0):
    model_path = ensure_optimized_model()
    if model_path is None:
        print("Error: no optimized model found. Run compile_qai.py or provide the model bundle.")
        return
    
    print("Loading model onto Hexagon NPU via QNN Execution Provider...")
    session_options = ort.SessionOptions()
    
    # Prioritize QNN (Qualcomm Neural Network) for NPU, fallback to CPU
    providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
    except Exception as e:
        print(f"QNN provider unavailable ({e}); falling back to CPU.")
        try:
            session = ort.InferenceSession(model_path, sess_options=session_options, providers=['CPUExecutionProvider'])
        except Exception as cpu_error:
            print(f"Failed to load ONNX model: {cpu_error}")
            return

    input_name = session.get_inputs()[0].name
    
    # Use 0 for webcam, or path to a dashcam .mp4 file
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting real-time ADAS inference...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        start_time = time.time()
        
        # Preprocess frame (Simulating cropping a traffic sign ROI)
        # Resize to GTSRB 32x32 standard
        img_resized = cv2.resize(frame, (32, 32))
        
        # Convert HWC to CHW format and normalize
        img_normalized = np.expand_dims(img_resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
        img_normalized = (img_normalized - 0.5) / 0.5
        
        # Run NPU Inference
        inputs = {input_name: img_normalized}
        predictions = session.run(None, inputs)
        
        class_id = np.argmax(predictions[0])
        inference_time = (time.time() - start_time) * 1000
        
        # Overlay results on the frame
        cv2.putText(frame, f"Class ID: {class_id}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {inference_time:.1f}ms", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('SnapSign ADAS NPU Inference', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=0, help="Webcam index or path to a video file")
    args = parser.parse_args()
    source = int(args.video) if str(args.video).isdigit() else args.video
    run_npu_inference(source)