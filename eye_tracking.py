import cv2
import mediapipe as mp

def track_eyes(frame):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Convert frame to RGB (for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        # Get eye landmarks
        landmarks = results.multi_face_landmarks[0]
        left_eye = landmarks.landmark[133]  # Left eye center
        right_eye = landmarks.landmark[362]  # Right eye center
        
        # Average the eye positions to get cursor position
        cursor_x = int((left_eye.x + right_eye.x) * frame.shape[1] / 2)
        cursor_y = int((left_eye.y + right_eye.y) * frame.shape[0] / 2)
        return (cursor_x, cursor_y)
    return (0, 0)
