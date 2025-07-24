"""
Hand Tracking Module for Facial Expression-based Artboard
Provides precise finger tracking for drawing control
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math


class HandTracker:
    """Advanced hand tracking with gesture recognition for drawing"""
    
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Drawing control
        self.drawing_finger = 8  # Index finger tip
        self.thumb_tip = 4
        self.previous_position = None
        self.is_drawing = False
        
        # Smoothing
        self.position_buffer = deque(maxlen=5)
        self.smoothing_factor = 0.7
        
        # Gesture detection
        self.pinch_threshold = 0.05
        self.gesture_buffer = deque(maxlen=3)
        
    def process_frame(self, frame):
        """
        Process frame and return hand tracking results
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            dict: Hand tracking results with position and gestures
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        tracking_data = {
            'landmarks_found': False,
            'finger_position': None,
            'is_drawing': False,
            'gesture': 'none',
            'annotated_frame': frame.copy()
        }
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                tracking_data['annotated_frame'],
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Get finger positions
            finger_pos = self._get_finger_position(hand_landmarks, frame.shape)
            drawing_state = self._detect_drawing_gesture(hand_landmarks)
            gesture = self._recognize_gesture(hand_landmarks)
            
            tracking_data.update({
                'landmarks_found': True,
                'finger_position': finger_pos,
                'is_drawing': drawing_state,
                'gesture': gesture
            })
        
        return tracking_data
    
    def _get_finger_position(self, landmarks, frame_shape):
        """Get smoothed finger position for drawing"""
        h, w = frame_shape[:2]
        
        # Get index finger tip coordinates
        finger_tip = landmarks.landmark[self.drawing_finger]
        x = int(finger_tip.x * w)
        y = int(finger_tip.y * h)
        
        # Add to buffer for smoothing
        self.position_buffer.append((x, y))
        
        # Apply smoothing
        if len(self.position_buffer) > 1:
            # Weighted average with more weight on recent positions
            weights = np.linspace(0.3, 1.0, len(self.position_buffer))
            weights = weights / weights.sum()
            
            positions = np.array(list(self.position_buffer))
            smoothed = np.sum(positions * weights[:, np.newaxis], axis=0)
            
            return int(smoothed[0]), int(smoothed[1])
        
        return x, y
    
    def _detect_drawing_gesture(self, landmarks):
        """Detect if user wants to draw (pinch gesture)"""
        # Get thumb and index finger positions
        thumb = landmarks.landmark[self.thumb_tip]
        index = landmarks.landmark[self.drawing_finger]
        
        # Calculate distance
        distance = math.sqrt(
            (thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2
        )
        
        # Add to gesture buffer for stability
        is_pinching = distance < self.pinch_threshold
        self.gesture_buffer.append(is_pinching)
        
        # Require majority vote for drawing state
        if len(self.gesture_buffer) >= 2:
            return sum(self.gesture_buffer) > len(self.gesture_buffer) // 2
        
        return is_pinching
    
    def _recognize_gesture(self, landmarks):
        """Recognize hand gestures for different drawing modes"""
        # Get finger tip positions
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_states = []
        
        for tip_id in finger_tips:
            tip = landmarks.landmark[tip_id]
            # For thumb, compare x-coordinate; for others, compare y-coordinate
            if tip_id == 4:  # Thumb
                base = landmarks.landmark[3]
                is_up = tip.x > base.x
            else:
                base = landmarks.landmark[tip_id - 2]
                is_up = tip.y < base.y
            
            finger_states.append(is_up)
        
        # Gesture recognition
        if finger_states == [True, True, False, False, False]:  # Thumb + Index
            return 'draw'
        elif finger_states == [False, True, False, False, False]:  # Index only
            return 'point'
        elif finger_states == [True, True, True, False, False]:  # Three fingers
            return 'erase'
        elif all(finger_states):  # All fingers up
            return 'clear'
        else:
            return 'none'
    
    def get_drawing_cursor(self, frame_width, frame_height, canvas_width, canvas_height):
        """Convert camera coordinates to canvas coordinates"""
        if not self.position_buffer:
            return None
        
        cam_x, cam_y = self.position_buffer[-1]
        
        # Convert coordinates (flip horizontally for mirror effect)
        canvas_x = int((1 - cam_x / frame_width) * canvas_width)
        canvas_y = int((cam_y / frame_height) * canvas_height)
        
        return canvas_x, canvas_y
    
    def close(self):
        """Clean up resources"""
        if self.hands:
            self.hands.close()


def test_hand_tracking():
    """Test function for hand tracking"""
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    print("Testing Hand Tracking - Press 'q' to quit")
    print("Gestures:")
    print("- Pinch (thumb + index): Draw")
    print("- Index finger only: Point")
    print("- Three fingers: Erase mode")
    print("- Open hand: Clear canvas")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        tracking_data = tracker.process_frame(frame)
        
        # Display tracking info
        if tracking_data['landmarks_found']:
            pos = tracking_data['finger_position']
            drawing = tracking_data['is_drawing']
            gesture = tracking_data['gesture']
            
            # Draw cursor
            if pos:
                color = (0, 255, 0) if drawing else (255, 0, 0)
                cv2.circle(tracking_data['annotated_frame'], pos, 10, color, -1)
            
            # Display info
            info_text = f"Gesture: {gesture} | Drawing: {drawing}"
            cv2.putText(tracking_data['annotated_frame'], info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Hand Tracking Test', tracking_data['annotated_frame'])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    test_hand_tracking()
