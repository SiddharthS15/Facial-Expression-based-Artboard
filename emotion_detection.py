"""
Emotion Detection Module for Facial Expression-based Artboard
Maps facial emotions to Bharatanatyam mudras for artistic expression
"""

import cv2
import numpy as np
try:
    from fer import FER
except ImportError:
    print("Warning: FER library not available. Emotion detection will use dummy mode.")
    FER = None
from collections import deque
import time


class EmotionDetector:
    """Advanced emotion detector with smoothing and stability features"""
    
    def __init__(self):
        # Initialize FER with MTCNN for better face detection
        if FER is not None:
            try:
                self.detector = FER(mtcnn=True)
                self.fer_available = True
            except Exception as e:
                print(f"Warning: Could not initialize FER detector: {e}")
                self.detector = None
                self.fer_available = False
        else:
            self.detector = None
            self.fer_available = False
        
        # Emotion to Bharatanatyam mudra mapping
        self.mudra_dict = {
            'angry': "(Roudra)",      # Fierce/Anger - Red energy
            'fear': "(Bhayanaka)",    # Fearful - Dark emotions  
            'happy': "(Hasya)",       # Joyful - Bright colors
            'neutral': "(Shanta)",    # Peaceful - Calm colors
            'sad': "(Karuna)",        # Compassionate/Sad - Soft tones
            'surprise': "(Adbhuta)"   # Wonder - Vibrant colors
        }
        
        # Smoothing and stability parameters
        self.buffer_size = 7
        self.emotion_buffer = deque(maxlen=self.buffer_size)
        self.last_emotion_change = time.time()
        self.emotion_change_cooldown = 1.2  # Prevent rapid changes
        
        # Performance optimization
        self.last_frame_time = time.time()
        self.process_every_n_frames = 3
        self.frame_counter = 0
        
    def detect_emotion(self, frame):
        """
        Detect emotion from frame with smoothing and stability
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            str: Detected mudra/emotion string
        """
        current_time = time.time()
        self.frame_counter += 1
        
        # Skip frames for performance
        if self.frame_counter % self.process_every_n_frames != 0:
            return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
        
        # Enforce cooldown to prevent rapid changes
        if current_time - self.last_emotion_change < self.emotion_change_cooldown:
            return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
        
        try:
            # If FER is not available, use dummy detection
            if not self.fer_available:
                # Return a random emotion for demonstration
                import random
                emotions = list(self.mudra_dict.values())
                return random.choice(emotions)
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
            
            # Detect emotion
            result = self.detector.top_emotion(small_frame)
            
            if result and result[0] in self.mudra_dict:
                detected_emotion = self.mudra_dict[result[0]]
            else:
                detected_emotion = "(Shanta)"
            
            # Add to buffer
            self.emotion_buffer.append(detected_emotion)
            
            # Use weighted voting for stability
            if len(self.emotion_buffer) >= 3:
                most_common = self._get_stable_emotion()
                
                # Only change if we have strong confidence
                if most_common != (self.emotion_buffer[-2] if len(self.emotion_buffer) > 1 else None):
                    self.last_emotion_change = current_time
                
                return most_common
            
            return detected_emotion
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "(Shanta)"
    
    def _get_stable_emotion(self):
        """Get most stable emotion using weighted voting"""
        if not self.emotion_buffer:
            return "(Shanta)"
        
        # Count emotions with recency weighting
        emotion_weights = {}
        weights = np.linspace(0.5, 1.0, len(self.emotion_buffer))
        
        for emotion, weight in zip(self.emotion_buffer, weights):
            emotion_weights[emotion] = emotion_weights.get(emotion, 0) + weight
        
        # Return most weighted emotion
        return max(emotion_weights.items(), key=lambda x: x[1])[0]
    
    def get_emotion_info(self, emotion):
        """Get additional info about the emotion/mudra"""
        info_dict = {
            "(Roudra)": {"color": "red", "tool": "brush", "description": "Fierce/Anger"},
            "(Bhayanaka)": {"color": "purple", "tool": "pencil", "description": "Fear/Terror"},
            "(Hasya)": {"color": "yellow", "tool": "marker", "description": "Joy/Laughter"},
            "(Shanta)": {"color": "blue", "tool": "pencil", "description": "Peace/Calm"},
            "(Karuna)": {"color": "green", "tool": "brush", "description": "Compassion/Sadness"},
            "(Adbhuta)": {"color": "orange", "tool": "marker", "description": "Wonder/Surprise"}
        }
        return info_dict.get(emotion, {"color": "black", "tool": "pencil", "description": "Unknown"})


def test_emotion_detection():
    """Test function for emotion detection"""
    detector = EmotionDetector()
    cap = cv2.VideoCapture(0)
    
    print("Testing Emotion Detection - Press 'q' to quit")
    print("Make different facial expressions to see mudra detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        emotion = detector.detect_emotion(frame)
        emotion_info = detector.get_emotion_info(emotion)
        
        # Display emotion info
        cv2.putText(frame, f"Mudra: {emotion}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Description: {emotion_info['description']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {emotion_info['color']}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Emotion Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_emotion_detection()
