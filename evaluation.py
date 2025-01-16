from app import FaceRecognitionSystem
import cv2
import time
import torch

def calculate_inference_time(frs, image_path):
    try:
        # Read the image using OpenCV
        frame = cv2.imread(image_path)

        # Measure inference time
        start_time = time.time()
        _, embedding2d, embedding3d = frs.process_frame_for_inference(frame)
        if embedding2d is not None and embedding3d is not None:
            _, status = frs.inference_mode(embedding2d, embedding3d)
        end_time = time.time()

        inference_time_ms = (end_time - start_time) * 1000
        return inference_time_ms

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # Use CPU for inference
    frs = FaceRecognitionSystem(device)
    
    image_path = "captured_image.jpg"  # Replace with your image path
    inference_time = calculate_inference_time(frs, image_path)

    if inference_time is not None:
        print(f"Inference time: {inference_time:.2f} ms")