import cv2
import numpy as np
#from tensorflow.keras.models import load_model

# Load your mudra detection model
model = load_model('models/face_mudra.hy')

def detect_mudra(frame):
    # Preprocess frame for model input
    # Add preprocessing like resizing, normalizing, etc.
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize

    # Predict mudra
    prediction = model.predict(img)
    mudra = np.argmax(prediction)

    if mudra == 0:
        return 'circle'
    elif mudra == 1:
        return 'line'
    else:
        return 'pencil'
