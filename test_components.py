"""
Test script for individual components of the Facial Expression-based Artboard
Run this to test each component separately
"""

import sys
import cv2

def test_camera():
    """Test if camera is working"""
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not found or cannot be opened")
        return False
    
    ret, frame = cap.read()
    if ret:
        print("✅ Camera working properly")
        print(f"Frame size: {frame.shape}")
    else:
        print("❌ Cannot read from camera")
        return False
    
    cap.release()
    return True

def test_emotion_detection():
    """Test emotion detection module"""
    print("\nTesting emotion detection...")
    try:
        from emotion_detection import EmotionDetector
        detector = EmotionDetector()
        print("✅ Emotion detection module loaded successfully")
        
        # Test with a dummy frame
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emotion = detector.detect_emotion(dummy_frame)
        print(f"✅ Emotion detection working. Default emotion: {emotion}")
        return True
    except Exception as e:
        print(f"❌ Emotion detection failed: {e}")
        return False

def test_hand_tracking():
    """Test hand tracking module"""
    print("\nTesting hand tracking...")
    try:
        from hand_tracking import HandTracker
        tracker = HandTracker()
        print("✅ Hand tracking module loaded successfully")
        
        # Test with dummy frame
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process_frame(dummy_frame)
        print(f"✅ Hand tracking working. Result keys: {list(result.keys())}")
        tracker.close()
        return True
    except Exception as e:
        print(f"❌ Hand tracking failed: {e}")
        return False

def test_drawing_canvas():
    """Test drawing canvas module"""
    print("\nTesting drawing canvas...")
    try:
        import tkinter as tk
        from drawing_canvas import DrawingCanvas
        
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        canvas = DrawingCanvas(root)
        print("✅ Drawing canvas module loaded successfully")
        
        # Test basic functionality
        canvas.start_drawing(100, 100)
        canvas.draw_to(200, 200)
        canvas.end_drawing()
        print("✅ Drawing functionality working")
        
        root.destroy()
        return True
    except Exception as e:
        print(f"❌ Drawing canvas failed: {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("Testing dependencies...")
    
    dependencies = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("fer", "fer"),
        ("tensorflow", "tensorflow")
    ]
    
    all_good = True
    for package_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Not installed")
            all_good = False
    
    return all_good

def run_interactive_test():
    """Run an interactive test with camera"""
    print("\n🎥 Starting interactive test (press 'q' to quit)...")
    
    try:
        from emotion_detection import EmotionDetector
        from hand_tracking import HandTracker
        
        detector = EmotionDetector()
        tracker = HandTracker()
        cap = cv2.VideoCapture(0)
        
        print("Make facial expressions and hand gestures!")
        print("The system will show detected emotions and hand states")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Test emotion detection
            emotion = detector.detect_emotion(frame)
            emotion_info = detector.get_emotion_info(emotion)
            
            # Test hand tracking
            hand_data = tracker.process_frame(frame)
            
            # Display results
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Color: {emotion_info['color']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if hand_data['landmarks_found']:
                cv2.putText(frame, f"Hand: {hand_data['gesture']}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Drawing: {hand_data['is_drawing']}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Interactive Test', hand_data['annotated_frame'])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("✅ Interactive test completed successfully!")
        
    except Exception as e:
        print(f"❌ Interactive test failed: {e}")

def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 FACIAL EXPRESSION ARTBOARD - COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Camera", test_camera),
        ("Emotion Detection", test_emotion_detection),
        ("Hand Tracking", test_hand_tracking),
        ("Drawing Canvas", test_drawing_canvas)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The application should work properly.")
        
        response = input("\nWould you like to run an interactive test? (y/n): ")
        if response.lower() == 'y':
            run_interactive_test()
        
        print("\nYou can now run the main application with:")
        print("python main.py")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
