# Facial Expression-based Artboard

An innovative digital art application that combines facial emotion recognition and hand gesture tracking to create an intuitive and expressive drawing experience. The application maps your facial expressions to different drawing tools and colors inspired by Bharatanatyam mudras, while your hand gestures control the drawing actions.

## 🌟 Features

### Emotion-Driven Art Creation
- **Real-time Emotion Detection**: Uses FER (Facial Emotion Recognition) to detect your emotions
- **Bharatanatyam Mudra Mapping**: Maps emotions to classical Indian dance expressions:
  - **(Roudra)** - Anger → Red brush, bold strokes
  - **(Bhayanaka)** - Fear → Purple pencil, thin lines  
  - **(Hasya)** - Joy → Yellow marker, medium strokes
  - **(Shanta)** - Peace → Blue pencil, fine lines
  - **(Karuna)** - Compassion/Sadness → Green brush, soft strokes
  - **(Adbhuta)** - Wonder/Surprise → Orange marker, bold strokes

### Advanced Hand Tracking
- **Gesture Recognition**: Multiple hand gestures for different functions:
  - **Pinch** (thumb + index): Draw on canvas
  - **Point** (index only): Move cursor without drawing
  - **Three Fingers**: Switch to eraser mode
  - **Open Hand**: Clear canvas
- **Smooth Drawing**: Advanced smoothing algorithms for natural-looking strokes
- **Pressure Simulation**: Drawing speed affects stroke thickness

### Professional Drawing Tools
- **Multiple Brush Types**: Pencil, brush, marker, eraser
- **Pressure Sensitivity**: Simulated pressure based on drawing speed
- **Line Smoothing**: Bezier curve smoothing for professional results
- **Undo/Redo System**: Full action history management
- **Save Functionality**: Export your artwork as PNG/JPEG

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam for emotion and hand detection
- Good lighting for optimal detection

### Setup Instructions

1. **Clone or download the project**
   ```bash
   git clone <your-repo-url>
   cd Facial-Expression-based-Artboard
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\\Scripts\\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the installation**
   ```bash
   python test_components.py
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## 🎮 How to Use

### Getting Started
1. **Launch the application**: Run `python main.py`
2. **Allow camera access**: Grant permission when prompted
3. **Position yourself**: Sit about 2-3 feet from the camera with good lighting
4. **Start creating**: Make facial expressions and use hand gestures to draw!

### Emotion Control
- **Make facial expressions** to automatically change drawing tools and colors
- **Happy expressions** → Bright colors and markers
- **Calm expressions** → Blue tones and fine pencils
- **Angry expressions** → Red colors and bold brushes
- **Surprised expressions** → Vibrant oranges and markers

### Hand Gestures
- **Drawing**: Pinch your thumb and index finger together, then move to draw
- **Moving**: Point with your index finger to move the cursor without drawing
- **Erasing**: Show three fingers to switch to eraser mode
- **Clearing**: Open your hand (all five fingers) to clear the canvas

### Manual Controls
- **Toggle Features**: Enable/disable emotion control and hand tracking
- **Manual Tools**: Override automatic settings with manual tool selection
- **Color Picker**: Choose custom colors
- **Brush Size**: Adjust brush thickness
- **Save/Load**: Save your artwork or load previous drawings

## 📁 Project Structure

```
Facial-Expression-based-Artboard/
├── main.py                 # Main application entry point
├── emotion_detection.py    # Emotion recognition module
├── hand_tracking.py        # Hand gesture tracking module
├── drawing_canvas.py       # Advanced drawing canvas
├── test_components.py      # Component testing script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Technical Details

### Components
- **Emotion Detection**: Uses FER library with MTCNN face detection
- **Hand Tracking**: MediaPipe Hands for precise finger tracking
- **Drawing Engine**: Custom Tkinter canvas with advanced features
- **UI Framework**: Tkinter with ttk for modern appearance

### Key Algorithms
- **Emotion Smoothing**: Weighted voting system prevents rapid changes
- **Hand Stabilization**: Deque-based position smoothing
- **Pressure Simulation**: Speed-based stroke width variation
- **Gesture Recognition**: Multi-finger state analysis

### Performance Optimizations
- **Frame Skipping**: Processes every 3rd frame for emotion detection
- **Threaded Processing**: Separate threads for UI and camera processing
- **Efficient Drawing**: Optimized canvas operations

## 🎨 Bharatanatyam Integration

This application draws inspiration from Bharatanatyam, a classical Indian dance form known for its expressive facial expressions (mukhabhinaya) and hand gestures (hastas). Each emotion detected is mapped to a traditional rasa (emotional state):

- **Roudra Rasa** (Fury) - Bold, aggressive strokes in red
- **Bhayanaka Rasa** (Terror) - Sharp, thin lines in dark colors  
- **Hasya Rasa** (Mirth) - Flowing, cheerful strokes in bright colors
- **Shanta Rasa** (Peace) - Calm, controlled lines in cool colors
- **Karuna Rasa** (Compassion) - Soft, gentle strokes in earth tones
- **Adbhuta Rasa** (Wonder) - Dynamic, surprising marks in vibrant colors

## 🐛 Troubleshooting

### Common Issues

**Camera not working:**
- Ensure your webcam is connected and not used by other applications
- Check camera permissions in your system settings
- Try running `python test_components.py` to diagnose issues

**Slow performance:**
- Ensure good lighting for better face detection
- Close other applications using the camera
- Lower the camera resolution in the code if needed

**Emotion detection not working:**
- Make sure your face is clearly visible and well-lit
- Try exaggerated facial expressions
- Check if the FER library installed correctly

**Hand tracking issues:**
- Ensure your hand is clearly visible against the background
- Try different hand positions and lighting
- Calibrate the pinch sensitivity in the code if needed

### Performance Tips
- Use good lighting (natural light works best)
- Keep a plain background behind you
- Position yourself 2-3 feet from the camera
- Make clear, deliberate facial expressions
- Keep hand movements smooth and controlled

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- **Bug Reports**: Report issues with detailed steps to reproduce
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with new features or fixes
- **Documentation**: Improve documentation and examples
- **Testing**: Test on different systems and provide feedback

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe** - Google's framework for hand tracking
- **FER** - Facial Emotion Recognition library
- **OpenCV** - Computer vision library
- **Bharatanatyam** - Classical Indian dance form for emotional inspiration

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Run `python test_components.py` to diagnose problems
3. Create an issue on the project repository
4. Provide system details and error messages for faster help

---

**Happy Creating! 🎨✨**

Transform your emotions and gestures into beautiful digital art with this innovative application that bridges traditional artistic expression with modern technology.
