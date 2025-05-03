import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from mtcnn import MTCNN  # Install with: pip install mtcnn

# Initialize MTCNN for face detection
detector = MTCNN()

def detect_faces(frame):
    """
    Detect faces in a frame using MTCNN.
    Args:
        frame (numpy.ndarray): Input frame (BGR format).
    Returns:
        list: List of cropped face images (RGB format).
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    faces = detector.detect_faces(rgb_frame)
    
    cropped_faces = []
    for face in faces:
        x, y, width, height = face['box']
        # Ensure the bounding box is within the frame dimensions
        x, y = max(x, 0), max(y, 0)
        width, height = min(width, rgb_frame.shape[1] - x), min(height, rgb_frame.shape[0] - y)
        face_img = rgb_frame[y:y+height, x:x+width]
        cropped_faces.append(face_img)
    
    return cropped_faces

def extract_faces_from_video(video_path, sequence_length=30):
    """
    Extract faces from a video and return a list of face frames.
    Args:
        video_path (str): Path to the input video.
        sequence_length (int): Number of frames to extract.
    Returns:
        list: List of face frames (PIL images).
    """
    vidObj = cv2.VideoCapture(video_path)
    success, frame = vidObj.read()
    frames = []

    while success:
        faces = detect_faces(frame)  # Detect faces in the frame
        if faces:
            for face in faces:
                face_img = Image.fromarray(face)  # Convert to PIL image
                frames.append(face_img)
                if len(frames) >= sequence_length:
                    break
        success, frame = vidObj.read()

    vidObj.release()

    # If fewer than `sequence_length` faces are detected, duplicate the last face
    while len(frames) < sequence_length:
        frames.append(frames[-1])

    return frames[:sequence_length]  # Return exactly `sequence_length` frames

def preprocess_frames(frames, transform):
    """
    Preprocess frames for the model.
    Args:
        frames (list): List of face frames (PIL images).
        transform: PyTorch transforms to apply.
    Returns:
        torch.Tensor: Stacked and transformed frames.
    """
    transformed_frames = [transform(frame) for frame in frames]
    return torch.stack(transformed_frames)  # Shape: (sequence_length, 3, H, W)

def get_transforms():
    """
    Define the preprocessing transforms for the frames.
    Returns:
        transforms.Compose: PyTorch transforms.
    """
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])