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
        self.detector = FER()
        self.mudra_dict = {
            'angry': "(Roudra)",
            'fear': "(Bhayanaka)",
            'happy': "(Hasya)",
            'neutral': "(Shanta)",
            'sad': "(Karuna)",
            'surprise': "(Adbhuta)"
        }
        self.buffer_size = 10
        self.emotion_buffer = []
        self.last_emotion_change = time.time()
        self.emotion_change_cooldown = 1.0

    def detect_emotion(self, frame):
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_emotion_change < self.emotion_change_cooldown:
            return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"
        
        # Process frame for emotion detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.top_emotion(rgb_frame)
        
        if result and result[0] in self.mudra_dict:
            detected_emotion = self.mudra_dict[result[0]]
        else:
            detected_emotion = "(Shanta)"
        
        # Update emotion buffer
        self.emotion_buffer.append(detected_emotion)
        if len(self.emotion_buffer) > self.buffer_size:
            self.emotion_buffer.pop(0)
        
        # Check for stable emotion
        most_common = max(set(self.emotion_buffer), key=self.emotion_buffer.count)
        if self.emotion_buffer.count(most_common) >= self.buffer_size * 0.6:
            if not self.emotion_buffer or most_common != self.emotion_buffer[-1]:
                self.last_emotion_change = current_time
            return most_common
        
        return self.emotion_buffer[-1] if self.emotion_buffer else "(Shanta)"

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion-Driven Paint App")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        self.cursor_id = None
        self.last_tool_update = time.time()
        self.last_emotion_change = time.time()
        
        # Drawing smoothing parameters
        self.point_buffer = deque(maxlen=3)
        self.last_draw_time = time.time()
        self.min_distance = 3
        self.smoothing_factor = 0.6
        self.pressure_simulation = True
        self.max_speed = 1000
        self.min_width_factor = 0.7

        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize video capture
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise ValueError("Could not open camera")
        except Exception as e:
            messagebox.showerror("Error", f"Could not initialize camera: {str(e)}")
            sys.exit(1)

        # Drawing state
        self.drawing = False
        self.previous_position = None
        self.mouse_drawing = False
        self.running = True

        # UI Variables
        self.stroke_size = tk.IntVar(value=2)
        self.stroke_color = tk.StringVar(value="black")
        self.current_tool = "pencil"
        self.iris_tracking_enabled = BooleanVar(value=False)

        # Emotion tracking
        self.emotion_detector = EmotionDetector()
        self.emotion = "(Shanta)"
        self.previous_emotion = None
        
        # Emotion-tool mapping
        self.emotion_tool_mapping = {
            "(Shanta)": ("pencil", "black"),
            "(Karuna)": ("eraser", "white"),
            "(Hasya)": ("pencil", "blue"),
            "(Bhayanaka)": ("pencil", "red"),
            "(Roudra)": ("pencil", "green"),
            "(Adbhuta)": ("pencil", "purple")
        }

        # Setup UI and start video thread
        self.setup_ui()
        self.setup_canvas_bindings()
        self.video_thread = Thread(target=self.update_video, daemon=True)
        self.video_thread.start()

        # Bind cleanup to window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left sidebar
        left_frame = tk.Frame(main_frame, bg="#e0e0e0", width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        self.setup_toolbar(left_frame)
        self.setup_color_palette(left_frame)

        # Iris tracking checkbox
        self.iris_checkbox = tk.Checkbutton(
            left_frame,
            text="Enable Iris Tracking",
            variable=self.iris_tracking_enabled,
            command=self.toggle_iris_tracking,
            bg="#e0e0e0"
        )
        self.iris_checkbox.pack(pady=10)

        # Main content area
        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = tk.Canvas(
            right_frame,
            bg="white",
            relief="ridge",
            bd=2,
            width=900,
            height=500
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Bottom frame for video and emotion
        bottom_frame = tk.Frame(right_frame, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.X)

        self.video_frame = tk.Label(
            bottom_frame,
            bg="#e0e0e0",
            relief="sunken",
            bd=2,
            width=500,
            height=500
        )
        self.video_frame.pack(side=tk.LEFT, padx=(0, 10))

        # Enhanced emotion display
        self.emotion_label = tk.Label(
            bottom_frame,
            text=f"Mudra: {self.emotion}",
            font=("Arial", 14),
            bg="#e0e0e0",
            relief="sunken",
            bd=2
        )
        self.emotion_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def setup_toolbar(self, parent):
        toolbar = tk.Frame(parent, bg="#e0e0e0")
        toolbar.pack(pady=(0, 10), fill=tk.X)

        buttons = [
            ("Clear", self.clear),
            ("Save", self.save_image)
        ]

        for text, command in buttons:
            tk.Button(
                toolbar,
                text=text,
                command=command,
                relief=tk.RAISED,
                bg="#d0d0d0"
            ).pack(fill=tk.X, pady=2)

        tk.Label(toolbar, text="Stroke Size", bg="#e0e0e0").pack(pady=(10, 0))
        tk.Scale(
            toolbar,
            variable=self.stroke_size,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            bg="#e0e0e0"
        ).pack(fill=tk.X)

    def setup_color_palette(self, parent):
        colors_frame = tk.LabelFrame(parent, text="Emotion Color Guide", bg="#e0e0e0", padx=5, pady=5)
        colors_frame.pack(fill=tk.X, padx=5, pady=5)

        mapping_frame = tk.Frame(colors_frame, bg="#e0e0e0")
        mapping_frame.pack(fill=tk.X)

        mappings = [
            ("Shanta (Neutral)", "black"),
            ("Karuna (Sad)", "white"),
            ("Hasya (Happy)", "blue"),
            ("Bhayanaka (Fear)", "red"),
            ("Roudra (Angry)", "green"),
            ("Adbhuta (Surprise)", "purple")
        ]

        for emotion, color in mappings:
            frame = tk.Frame(mapping_frame, bg="#e0e0e0")
            frame.pack(fill=tk.X, pady=2)
            
            color_sample = tk.Frame(frame, width=20, height=20, bg=color, relief="raised", bd=1)
            color_sample.pack(side=tk.LEFT, padx=5)
            color_sample.pack_propagate(False)
            
            tk.Label(
                frame,
                text=emotion,
                bg="#e0e0e0",
                font=("Arial", 10)
            ).pack(side=tk.LEFT, padx=5)

    def update_tool_based_on_emotion(self, emotion):
        current_time = time.time()
        if current_time - self.last_tool_update < 1:
            return
            
        self.last_tool_update = current_time
        if emotion in self.emotion_tool_mapping:
            tool, color = self.emotion_tool_mapping[emotion]
            self.current_tool = tool
            self.stroke_color.set(color)
            
            tool_text = "Eraser" if tool == "eraser" else f"{color.capitalize()} Pencil"
            self.emotion_label.config(
                text=f"Mudra: {emotion}\nTool: {tool_text}",
                fg=color if tool != "eraser" else "black"
            )

    def update_video(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (640, 480))
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                current_time = time.time()
                if current_time - self.last_emotion_change >= 0.5:
                    current_emotion = self.emotion_detector.detect_emotion(frame)
                    if current_emotion != self.emotion:
                        self.emotion = current_emotion
                        self.last_emotion_change = current_time
                        self.update_tool_based_on_emotion(current_emotion)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # Blink detection
                    left_eye_top = landmarks[145].y
                    left_eye_bottom = landmarks[159].y
                    blink_detected = (left_eye_bottom - left_eye_top) < 0.007

                    if blink_detected:
                        self.drawing = not self.drawing

                    if self.drawing and self.iris_tracking_enabled.get():
                        x, y = self.detect_iris_position(landmarks)
                        self._draw_line(x, y)
                else:
                    if self.iris_tracking_enabled.get():
                        self.previous_position = None

                # Update video display
                img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
                self.video_frame.configure(image=img)
                self.video_frame.image = img

                time.sleep(0.04)

            except Exception as e:
                print(f"Error in video processing: {str(e)}")
                continue

    def detect_iris_position(self, landmarks):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        iris = landmarks[474]  # Right iris center
        x = int(iris.x * canvas_width)
        y = int(iris.y * canvas_height)

        if self.cursor_id:
            self.canvas.delete(self.cursor_id)
        self.cursor_id = self.canvas.create_oval(
            x - 5, y - 5, x + 5, y + 5,
            fill="red" if self.current_tool == "eraser" else self.stroke_color.get(),
            outline=""
        )
        return x, y

    def _smooth_points(self, x, y):
        self.point_buffer.append((x, y))
        
        if len(self.point_buffer) < 2:
            return x, y
        
        points = list(self.point_buffer)
        smoothed_x = sum(p[0] for p in points) / len(points)
        smoothed_y = sum(p[1] for p in points) / len(points)
        
        final_x = x * (1 - self.smoothing_factor) + smoothed_x * self.smoothing_factor
        final_y = y * (1 - self.smoothing_factor) + smoothed_y * self.smoothing_factor
        
        return int(final_x), int(final_y)

    def _draw_line(self, x, y):
        current_time = time.time()
        if self.previous_position is None:
            self.previous_position = (x, y)
            self.last_draw_time = current_time
            return
            
        time_delta = current_time - self.last_draw_time
        if time_delta < 0.0001:
            return
            
        dx = x - self.previous_position[0]
        dy = y - self.previous_position[1]
        distance = (dx * dx + dy * dy) ** 0.5
        
        if distance < self.min_distance:
            return
            
        speed = distance / time_delta
        if self.pressure_simulation:
            speed_factor = min(speed / self.max_speed, 1.0)
            width_factor = 1.0 - (speed_factor * (1.0 - self.min_width_factor))
            current_width = self.stroke_size.get() * width_factor
        else:
            current_width = self.stroke_size.get()
        
        smooth_x, smooth_y = self._smooth_points(x, y)
        
        color = "white" if self.current_tool == "eraser" else self.stroke_color.get()
        
        self.canvas.create_line(
            self.previous_position[0],
            self.previous_position[1],
            smooth_x,
            smooth_y,
            fill=color,
            width=current_width,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=48
        )
        
        self.previous_position = (smooth_x, smooth_y)
        self.last_draw_time = current_time

    def setup_canvas_bindings(self):
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        if not self.iris_tracking_enabled.get():
            self.mouse_drawing = True
            self.previous_position = (event.x, event.y)

    def draw(self, event):
        if not self.iris_tracking_enabled.get() and self.mouse_drawing:
            self._draw_line(event.x, event.y)

    def stop_drawing(self, event):
        self.mouse_drawing = False
        self.previous_position = None

    def toggle_iris_tracking(self):
        if not self.iris_tracking_enabled.get():
            self.previous_position = None
        self.drawing = False

    def clear(self):
        self.canvas.delete("all")

    def save_image(self):
        try:
            file = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Files", "*.png")]
            )
            if file:
                x = self.canvas.winfo_rootx()
                y = self.canvas.winfo_rooty()
                width = self.canvas.winfo_width()
                height = self.canvas.winfo_height()
                ImageGrab.grab(bbox=(x, y, x + width, y + height)).save(file)
                messagebox.showinfo("Success", "Image saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = PaintApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}")
        sys.exit(1)