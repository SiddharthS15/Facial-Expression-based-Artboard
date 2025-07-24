"""
Drawing Canvas Module for Facial Expression-based Artboard
Provides advanced drawing capabilities with pressure simulation and effects
"""

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageGrab
import cv2
from collections import deque
import math


class DrawingCanvas:
    """Advanced drawing canvas with emotion-responsive features"""
    
    def __init__(self, parent, width=800, height=600):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Create canvas
        self.canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg='white',
            cursor='crosshair'
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Drawing state
        self.current_tool = 'pencil'
        self.current_color = 'black'
        self.brush_size = 5
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # Advanced drawing features
        self.pressure_simulation = True
        self.line_smoothing = True
        self.brush_opacity = 1.0
        
        # Stroke management
        self.stroke_buffer = deque(maxlen=10)
        self.stroke_points = []
        self.min_distance = 3
        
        # Emotion-based tool mapping
        self.emotion_tools = {
            "(Roudra)": {"color": "#FF4444", "size": 8, "tool": "brush"},      # Anger - Red, thick
            "(Bhayanaka)": {"color": "#8B00FF", "size": 4, "tool": "pencil"},  # Fear - Purple, thin
            "(Hasya)": {"color": "#FFD700", "size": 6, "tool": "marker"},      # Joy - Gold, medium
            "(Shanta)": {"color": "#4169E1", "size": 3, "tool": "pencil"},     # Peace - Blue, fine
            "(Karuna)": {"color": "#32CD32", "size": 5, "tool": "brush"},      # Compassion - Green
            "(Adbhuta)": {"color": "#FF6347", "size": 7, "tool": "marker"}     # Wonder - Orange, bold
        }
        
        # Undo/Redo system
        self.canvas_states = []
        self.state_index = -1
        self.max_states = 20
        
        self._save_state()
        
    def start_drawing(self, x, y):
        """Start a new drawing stroke"""
        self.is_drawing = True
        self.last_x = x
        self.last_y = y
        self.stroke_points = [(x, y)]
        
    def draw_to(self, x, y):
        """Continue drawing stroke to new position"""
        if not self.is_drawing or self.last_x is None or self.last_y is None:
            return
            
        # Calculate distance for pressure simulation
        distance = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
        
        if distance < self.min_distance:
            return
            
        # Add point to stroke
        self.stroke_points.append((x, y))
        
        # Draw line with current tool
        if self.current_tool == 'pencil':
            self._draw_pencil_line(self.last_x, self.last_y, x, y, distance)
        elif self.current_tool == 'brush':
            self._draw_brush_stroke(self.last_x, self.last_y, x, y, distance)
        elif self.current_tool == 'marker':
            self._draw_marker_line(self.last_x, self.last_y, x, y, distance)
        elif self.current_tool == 'eraser':
            self._erase_area(x, y)
        
        self.last_x = x
        self.last_y = y
        
    def end_drawing(self):
        """End current drawing stroke"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        if len(self.stroke_points) > 1:
            self._save_state()
            
    def _draw_pencil_line(self, x1, y1, x2, y2, distance):
        """Draw pencil-style line with pressure sensitivity"""
        if self.pressure_simulation:
            # Simulate pressure based on drawing speed
            speed = distance / 16.67  # Assuming ~60fps
            pressure = max(0.3, min(1.0, 1.0 - speed * 0.1))
            width = max(1, int(self.brush_size * pressure))
        else:
            width = self.brush_size
            
        # Draw main line
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=self.current_color,
            width=width,
            capstyle=tk.ROUND,
            smooth=self.line_smoothing
        )
        
        # Add texture for realistic pencil effect
        if width > 1:
            for i in range(2):
                offset = np.random.randint(-1, 2)
                alpha = max(0.3, self.brush_opacity - i * 0.2)
                color = self._adjust_color_alpha(self.current_color, alpha)
                self.canvas.create_line(
                    x1 + offset, y1 + offset, x2 + offset, y2 + offset,
                    fill=color,
                    width=max(1, width - i),
                    capstyle=tk.ROUND
                )
    
    def _draw_brush_stroke(self, x1, y1, x2, y2, distance):
        """Draw brush-style stroke with natural variation"""
        if self.pressure_simulation:
            speed = distance / 16.67
            pressure = max(0.5, min(1.2, 1.0 - speed * 0.05))
            width = max(2, int(self.brush_size * pressure))
        else:
            width = self.brush_size
            
        # Create brush effect with multiple overlapping lines
        for i in range(3):
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)
            alpha = max(0.4, self.brush_opacity - i * 0.15)
            color = self._adjust_color_alpha(self.current_color, alpha)
            
            self.canvas.create_line(
                x1 + offset_x, y1 + offset_y,
                x2 + offset_x, y2 + offset_y,
                fill=color,
                width=max(1, width - i),
                capstyle=tk.ROUND,
                smooth=True
            )
    
    def _draw_marker_line(self, x1, y1, x2, y2, distance):
        """Draw marker-style line with consistent width"""
        width = self.brush_size + 2  # Markers are typically thicker
        
        # Draw semi-transparent overlay effect
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=self.current_color,
            width=width,
            capstyle=tk.ROUND,
            smooth=True
        )
        
        # Add highlight effect
        highlight_color = self._lighten_color(self.current_color, 0.3)
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=highlight_color,
            width=max(1, width - 2),
            capstyle=tk.ROUND
        )
    
    def _erase_area(self, x, y):
        """Erase area around cursor"""
        erase_size = self.brush_size * 3
        self.canvas.create_oval(
            x - erase_size, y - erase_size,
            x + erase_size, y + erase_size,
            fill='white',
            outline='white'
        )
    
    def update_emotion_tool(self, emotion):
        """Update drawing tool based on detected emotion"""
        if emotion in self.emotion_tools:
            tool_config = self.emotion_tools[emotion]
            self.current_color = tool_config['color']
            self.brush_size = tool_config['size']
            self.current_tool = tool_config['tool']
            return True
        return False
    
    def set_tool(self, tool):
        """Manually set drawing tool"""
        valid_tools = ['pencil', 'brush', 'marker', 'eraser']
        if tool in valid_tools:
            self.current_tool = tool
    
    def set_color(self, color):
        """Set drawing color"""
        self.current_color = color
    
    def set_brush_size(self, size):
        """Set brush size"""
        self.brush_size = max(1, min(20, size))
    
    def clear_canvas(self):
        """Clear the entire canvas"""
        self.canvas.delete("all")
        self._save_state()
    
    def undo(self):
        """Undo last action"""
        if self.state_index > 0:
            self.state_index -= 1
            self._restore_state(self.state_index)
    
    def redo(self):
        """Redo last undone action"""
        if self.state_index < len(self.canvas_states) - 1:
            self.state_index += 1
            self._restore_state(self.state_index)
    
    def save_image(self, filename):
        """Save canvas as image"""
        try:
            # Get canvas coordinates
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()
            
            # Grab screenshot of canvas area
            ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def _save_state(self):
        """Save current canvas state for undo/redo"""
        try:
            # Remove future states if we're not at the end
            if self.state_index < len(self.canvas_states) - 1:
                self.canvas_states = self.canvas_states[:self.state_index + 1]
            
            # Create state representation (simplified - could be improved)
            state = {
                'items': list(self.canvas.find_all()),
                'tool': self.current_tool,
                'color': self.current_color,
                'size': self.brush_size
            }
            
            self.canvas_states.append(state)
            self.state_index = len(self.canvas_states) - 1
            
            # Limit number of states
            if len(self.canvas_states) > self.max_states:
                self.canvas_states.pop(0)
                self.state_index -= 1
                
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def _restore_state(self, index):
        """Restore canvas to saved state"""
        try:
            if 0 <= index < len(self.canvas_states):
                state = self.canvas_states[index]
                # Simplified restoration - in real implementation,
                # you'd need to store actual canvas data
                pass
        except Exception as e:
            print(f"Error restoring state: {e}")
    
    def _adjust_color_alpha(self, color, alpha):
        """Adjust color transparency (simplified)"""
        # In a real implementation, you'd convert to RGBA
        return color
    
    def _lighten_color(self, color, factor):
        """Lighten a color by given factor"""
        # Simplified - in real implementation, convert hex to RGB,
        # lighten, and convert back
        return color
    
    def get_canvas_data(self):
        """Get current canvas data for analysis"""
        return {
            'items': self.canvas.find_all(),
            'tool': self.current_tool,
            'color': self.current_color,
            'size': self.brush_size
        }
