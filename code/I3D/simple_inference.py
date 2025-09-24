import os
import cv2
import torch
import numpy as np
from pytorch_i3d import InceptionI3d
import torch.nn.functional as F


def load_rgb_frames_from_video(video_path, start=0, num=-1):
    """Load video frames from file."""
    print(f"Loading video from {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    frames = []
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    print(f"Video has {num} frames")
    
    for offset in range(num):
        success, img = vidcap.read()
        if not success:
            break
            
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Normalize pixel values to [-1, 1]
        img = (img / 255.) * 2 - 1
        
        frames.append(img)
    
    vidcap.release()
    print(f"Loaded {len(frames)} frames")
    return np.array(frames) if frames else None


def run_inference(video_path, model_weights, num_classes=2000):
    """Run inference on a single video using I3D model."""
    print("Setting up model...")
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    i3d.eval()
    
    # Load frames
    frames = load_rgb_frames_from_video(video_path)
    if frames is None:
        return None
    
    # Convert to tensor and add batch dimension
    # Shape: [1, T, H, W, C] -> needs to be [1, C, T, H, W] for the model
    frames_tensor = torch.FloatTensor(frames).unsqueeze(0)  # Add batch dim
    frames_tensor = frames_tensor.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
    
    print(f"Input tensor shape: {frames_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        per_frame_logits = i3d(frames_tensor)
    
    print(f"Output logits shape: {per_frame_logits.shape}")
    
    # Get predictions across all frames
    predictions = torch.max(per_frame_logits, dim=2)[0]
    probs = F.softmax(predictions, dim=1).cpu().numpy()[0]
    out_labels = np.argsort(predictions.cpu().numpy()[0])
    
    return out_labels, probs


def load_class_names(file_path='preprocess/wlasl_class_list.txt'):
    """Load class names from file."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None


if __name__ == "__main__":
    # Paths
    video_dir = '../../data/WLASL2000'
    model_weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    
    # Check for videos
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print("No video files found in the directory.")
        exit(1)
    
    # Select the first video
    video_path = os.path.join(video_dir, video_files[0])
    print(f"Selected video for inference: {video_path}")
    
    # Run inference
    predictions, probs = run_inference(video_path, model_weights)
    
    if predictions is None:
        print("Inference failed.")
        exit(1)
    
    # Show top predictions
    top_k = 10
    top_indices = predictions[-top_k:][::-1]  # Get top-k predictions in descending order
    
    # Load class names if available
    class_names = load_class_names()
    
    print("\nTop {} predictions:".format(top_k))
    for i, idx in enumerate(top_indices):
        confidence = probs[idx] * 100
        if class_names and idx < len(class_names):
            print(f"{i+1}. Class {idx}: {class_names[idx]} (Confidence: {confidence:.2f}%)")
        else:
            print(f"{i+1}. Class {idx} (Confidence: {confidence:.2f}%)")

