import cv2
import numpy as np
import onnxruntime as ort
import time

def run_npu_inference():
    model_path = "../models/snapsign_optimized.onnx"
    
    print("Loading model onto Hexagon NPU via QNN Execution Provider...")
    session_options = ort.SessionOptions()
    
    # Prioritize QNN (Qualcomm Neural Network) for NPU, fallback to CPU
    providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX model. Ensure {model_path} exists.")
        return

    input_name = session.get_inputs()[0].name
    
    # Use 0 for webcam, or path to a dashcam .mp4 file
    cap = cv2.VideoCapture(0) 
    
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
    run_npu_inference()