import cv2
import numpy as np
from mtcnn import MTCNN

class VideoProcessor:
    def __init__(self, target_size=(224, 224)):
        self.detector = MTCNN()
        self.target_size = target_size
        
    def extract_faces(self, video_path, frames_to_extract=30):
        """
        Extract faces from video and return a sequence of face images
        :param video_path: path to video file
        :param frames_to_extract: number of frames to extract (default 30)
        :return: numpy array of face images (seq_len, C, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, frames_to_extract, dtype=np.int32)
        
        faces = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i in frame_indices:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                detections = self.detector.detect_faces(rgb_frame)
                
                if len(detections) > 0:
                    # Get the face with highest confidence
                    best_face = max(detections, key=lambda x: x['confidence'])
                    x, y, w, h = best_face['box']
                    
                    # Expand the bounding box slightly
                    x, y = max(0, x-20), max(0, y-20)
                    w, h = min(w+40, frame.shape[1]-x), min(h+40, frame.shape[0]-y)
                    
                    # Extract face
                    face = rgb_frame[y:y+h, x:x+w]
                    
                    # Resize to target size
                    face = cv2.resize(face, self.target_size)
                    
                    # Normalize pixel values
                    face = face.astype(np.float32) / 255.0
                    
                    # Convert to CHW format
                    face = np.transpose(face, (2, 0, 1))
                    
                    faces.append(face)
        
        cap.release()
        
        if len(faces) == 0:
            return None
            
        # Convert list to numpy array and ensure we have exactly frames_to_extract
        if len(faces) > frames_to_extract:
            faces = faces[:frames_to_extract]
        elif len(faces) < frames_to_extract:
            # Repeat last frame if we don't have enough
            last_frame = faces[-1]
            while len(faces) < frames_to_extract:
                faces.append(last_frame)
                
        return np.stack(faces)