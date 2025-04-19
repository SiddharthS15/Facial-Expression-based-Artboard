import cv2
import mediapipe as mp
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, BooleanVar
from PIL import Image, ImageTk, ImageGrab
import numpy as np
from threading import Thread
import sys
from fer import FER
from collections import deque

class EmotionDetector:
    def __init__(self):
        # Initialize with a more efficient model configuration
        self.detector = FER(mtcnn=True)  # Using MTCNN for faster face detection
        self.mudra_dict = {
            'angry': "(Roudra)",
            'fear': "(Bhayanaka)",
            'happy': "(Hasya)",
            'neutral': "(Shanta)",
            'sad': "(Karuna)",
            'surprise': "(Adbhuta)"
        }
        self.buffer_size = 5  # Reduced buffer size for faster response
        self.emotion_buffer = deque(maxlen=self.buffer_size)
        self.last_emotion_change = time.time()
        self.emotion_change_cooldown = 0.5  # Reduced cooldown for more responsive changes
        self.last_frame_time = time.time()
        self.process_every_n_frames = 3  # Process every 3rd frame for better performance

    def detect_emotion(self, frame):
        current_time = time.time()
        
        # Skip frames to improve performance
        if current_time - self.last_frame_time < (1.0 / 30.0 * self.process_every_n_frames):
            return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
        
        self.last_frame_time = current_time
        
        if current_time - self.last_emotion_change < self.emotion_change_cooldown:
            return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        result = self.detector.top_emotion(small_frame)
        
        if result and result[0] in self.mudra_dict:
            detected_emotion = self.mudra_dict[result[0]]
        else:
            detected_emotion = "(Shanta)"
        
        self.emotion_buffer.append(detected_emotion)
        
        # Use weighted voting for more stable emotion detection
        emotions_count = {}
        weights = np.linspace(0.5, 1.0, len(self.emotion_buffer))
        
        for emotion, weight in zip(self.emotion_buffer, weights):
            emotions_count[emotion] = emotions_count.get(emotion, 0) + weight
        
        most_common = max(emotions_count.items(), key=lambda x: x[1])[0]
        
        if emotions_count[most_common] >= sum(weights) * 0.6:
            if not self.emotion_buffer or most_common != self.emotion_buffer[-1]:
                self.last_emotion_change = current_time
            return most_common
        
        return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
    def __init__(self):
        # Initialize with a more efficient model configuration
        self.detector = FER(mtcnn=True)  # Using MTCNN for faster face detection
        self.mudra_dict = {
            'angry': "(Roudra)",
            'fear': "(Bhayanaka)",
            'happy': "(Hasya)",
            'neutral': "(Shanta)",
            'sad': "(Karuna)",
            'surprise': "(Adbhuta)"
        }
        self.buffer_size = 5  # Reduced buffer size for faster response
        self.emotion_buffer = deque(maxlen=self.buffer_size)
        self.last_emotion_change = time.time()
        self.emotion_change_cooldown = 0.5  # Reduced cooldown for more responsive changes
        self.last_frame_time = time.time()
        self.process_every_n_frames = 3  # Process every 3rd frame for better performance

    def detect_emotion(self, frame):
        current_time = time.time()
        
        # Skip frames to improve performance
        if current_time - self.last_frame_time < (1.0 / 30.0 * self.process_every_n_frames):
            return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
        
        self.last_frame_time = current_time
        
        if current_time - self.last_emotion_change < self.emotion_change_cooldown:
            return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        result = self.detector.top_emotion(small_frame)
        
        if result and result[0] in self.mudra_dict:
            detected_emotion = self.mudra_dict[result[0]]
        else:
            detected_emotion = "(Shanta)"
        
        self.emotion_buffer.append(detected_emotion)
        
        # Use weighted voting for more stable emotion detection
        emotions_count = {}
        weights = np.linspace(0.5, 1.0, len(self.emotion_buffer))
        
        for emotion, weight in zip(self.emotion_buffer, weights):
            emotions_count[emotion] = emotions_count.get(emotion, 0) + weight
        
        most_common = max(emotions_count.items(), key=lambda x: x[1])[0]
        
        if emotions_count[most_common] >= sum(weights) * 0.6:
            if not self.emotion_buffer or most_common != self.emotion_buffer[-1]:
                self.last_emotion_change = current_time
            return most_common
        
        return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Emotion-Driven Paint App")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Enhanced drawing parameters
        self.cursor_id = None
        self.drawing = False
        self.point_buffer = deque(maxlen=5)  # Reduced buffer size for more responsive drawing
        self.last_draw_time = time.time()
        self.min_distance = 2  # Reduced minimum distance for smoother lines
        self.smoothing_factor = 0.7  # Increased smoothing factor
        self.pressure_simulation = True
        self.max_speed = 800  # Adjusted for better pressure sensitivity
        self.min_width_factor = 0.5
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize video capture with higher resolution
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                raise ValueError("Could not open camera")
        except Exception as e:
            messagebox.showerror("Error", f"Could not initialize camera: {str(e)}")
            sys.exit(1)
            
        # Drawing state
        self.previous_position = None
        self.running = True
        self.finger_drawing = False
        
        # Enhanced stroke interpolation
        self.stroke_points = deque(maxlen=3)
        self.last_stroke_time = time.time()
        
        # UI Variables
        self.stroke_size = tk.IntVar(value=2)
        self.stroke_color = tk.StringVar(value="black")
        self.current_tool = "pencil"
        self.finger_tracking_enabled = BooleanVar(value=True)
        
        # Emotion tracking
        self.emotion_detector = EmotionDetector()
        self.emotion = "(Shanta)"
        self.last_tool_update = time.time()
        
        # Enhanced emotion-tool mapping
        self.emotion_tool_mapping = {
            "(Shanta)": ("pencil", "black", 2),
            "(Karuna)": ("eraser", "white", 3),
            "(Hasya)": ("pencil", "blue", 2),
            "(Bhayanaka)": ("pencil", "red", 2),
            "(Roudra)": ("pencil", "green", 3),
            "(Adbhuta)": ("pencil", "purple", 2)
        }
        
        self.setup_ui()
        self.setup_canvas_bindings()
        self.video_thread = Thread(target=self.update_video, daemon=True)
        self.video_thread.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def detect_finger_position(self, hand_landmarks, frame_shape):
        # Get index finger tip coordinates
        index_finger_tip = hand_landmarks.landmark[8]  # MediaPipe index finger tip landmark
        
        # Convert normalized coordinates to canvas coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        x = int(index_finger_tip.x * canvas_width)
        y = int(index_finger_tip.y * canvas_height)
        
        # Get index finger distance from thumb for drawing control
        thumb_tip = hand_landmarks.landmark[4]  # MediaPipe thumb tip landmark
        distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + 
                   (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
        
        # Update drawing state based on finger pinch
        self.finger_drawing = distance < 0.1  # Adjust threshold as needed
        
        return x, y

    def _smooth_points(self, x, y):
        self.point_buffer.append((x, y))
        
        if len(self.point_buffer) < 2:
            return x, y
        
        # Enhanced Gaussian smoothing
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1][:len(self.point_buffer)])
        weights = weights / weights.sum()
        
        points = np.array(list(self.point_buffer))
        smoothed_point = np.sum(points * weights[:, np.newaxis], axis=0)
        
        # Apply additional bezier smoothing for curves
        if len(self.point_buffer) >= 3:
            p0 = np.array(self.point_buffer[-3])
            p1 = np.array(self.point_buffer[-2])
            p2 = np.array(self.point_buffer[-1])
            t = 0.5
            bezier_point = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            smoothed_point = smoothed_point * 0.7 + bezier_point * 0.3
        
        return int(smoothed_point[0]), int(smoothed_point[1])

    def _draw_line(self, x, y):
        current_time = time.time()
        if self.previous_position is None:
            self.previous_position = (x, y)
            self.last_draw_time = current_time
            return
        
        time_delta = current_time - self.last_draw_time
        if time_delta < 0.0001:
            return
        
        # Calculate drawing parameters
        dx = x - self.previous_position[0]
        dy = y - self.previous_position[1]
        distance = (dx * dx + dy * dy) ** 0.5
        
        if distance < self.min_distance:
            return
        
        # Enhanced pressure simulation
        speed = distance / time_delta
        if self.pressure_simulation:
            speed_factor = min(speed / self.max_speed, 1.0)
            width_factor = 1.0 - (speed_factor * (1.0 - self.min_width_factor))
            # Add slight randomness to line width for more natural look
            width_variation = np.random.uniform(0.95, 1.05)
            current_width = self.stroke_size.get() * width_factor * width_variation
        else:
            current_width = self.stroke_size.get()
        
        # Get smoothed coordinates
        smooth_x, smooth_y = self._smooth_points(x, y)
        
        # Draw the line
        color = "white" if self.current_tool == "eraser" else self.stroke_color.get()
        
        # Create multiple semi-transparent lines for more natural look
        alpha = 0.3
        for i in range(3):
            offset = i - 1
            self.canvas.create_line(
                self.previous_position[0] + offset,
                self.previous_position[1] + offset,
                smooth_x + offset,
                smooth_y + offset,
                fill=color,
                width=current_width * (1 - i*0.1),
                capstyle=tk.ROUND,
                smooth=True,
                splinesteps=64
            )
        
        self.previous_position = (smooth_x, smooth_y)
        self.last_draw_time = current_time

    def update_video(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hand landmarks
                if self.finger_tracking_enabled.get():
                    hand_results = self.hands.process(rgb_frame)
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # Draw hand landmarks for visual feedback
                            self.mp_drawing.draw_landmarks(
                                rgb_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS
                            )
                            
                            # Get finger position and update drawing
                            x, y = self.detect_finger_position(hand_landmarks, frame.shape)
                            if self.finger_drawing:
                                self._draw_line(x, y)
                            else:
                                self.previous_position = None
                
                # Process emotions
                current_time = time.time()
                if current_time - self.last_tool_update >= 0.5:
                    current_emotion = self.emotion_detector.detect_emotion(frame)
                    if current_emotion != self.emotion:
                        self.emotion = current_emotion
                        self.update_tool_based_on_emotion(current_emotion)
                        self.last_tool_update = current_time
                
                # Update video display
                img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
                self.video_frame.configure(image=img)
                self.video_frame.image = img
                
                time.sleep(0.016)  # Cap at ~60 FPS

            except Exception as e:
                print(f"Error in video processing: {str(e)}")
                continue

    # ... (rest of the UI setup methods remain the same) ...

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        if self.hands:
            self.hands.close()
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = PaintApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}")
        sys.exit(1)