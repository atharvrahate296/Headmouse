import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque
from enum import Enum
import tkinter as tk
from PIL import Image, ImageTk

# ==================== CONFIGURATION ====================
class GestureConfig:
    """Centralized configuration for gesture detection"""
    
    # Distance thresholds (in normalized coordinates)
    PINCH_THRESHOLD = 0.05  # Distance for pinch detection
    FINGER_EXTENDED_THRESHOLD = 0.1  # Threshold for finger extension
    
    # Debounce settings
    CLICK_COOLDOWN = 0.5  # Seconds between clicks
    GESTURE_HOLD_FRAMES = 3  # Frames to confirm gesture
    
    # Smoothing
    CURSOR_SMOOTHING = 7  # Higher = smoother but slower
    
    # Dead zone for cursor stability
    DEAD_ZONE = 15  # Pixels
    
    # Screen edge margin
    EDGE_MARGIN = 50

# ==================== GESTURE TYPES ====================
class Gesture(Enum):
    """All possible gestures"""
    NONE = 0
    OPEN_HAND = 1
    INDEX_THUMB_PINCH = 2  # Left click
    MIDDLE_THUMB_PINCH = 3  # Right click
    PEACE_SIGN = 4  # Scroll mode
    FIST = 5  # Lock cursor
    THUMBS_UP = 6  # Volume up (deprecated)
    THUMBS_DOWN = 7  # Volume down (deprecated)
    THUMB_PINKY_TOP = 8  # Volume up (NEW)
    THUMB_PINKY_BOTTOM = 9  # Volume down (NEW)

# ==================== SMOOTHING FILTER ====================
class SmoothingFilter:
    """Exponential moving average filter"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.alpha = 2 / (window_size + 1)
    
    def update(self, x, y):
        self.history.append((x, y))
        if len(self.history) == 1:
            return x, y
        
        prev_x, prev_y = self.history[-2]
        smooth_x = self.alpha * x + (1 - self.alpha) * prev_x
        smooth_y = self.alpha * y + (1 - self.alpha) * prev_y
        return smooth_x, smooth_y
    
    def reset(self):
        self.history.clear()

# ==================== GESTURE DETECTOR ====================
class HandGestureDetector:
    """Advanced hand gesture detection with proper debouncing"""
    
    # MediaPipe hand landmarks
    WRIST = 0
    THUMB_TIP = 4
    THUMB_IP = 3
    INDEX_TIP = 8
    INDEX_DIP = 7
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_DIP = 11
    MIDDLE_PIP = 10
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_MCP = 17
    
    def __init__(self, config=None):
        self.config = config or GestureConfig()
        
        # State tracking
        self.current_gesture = Gesture.NONE
        self.gesture_hold_counter = 0
        self.last_confirmed_gesture = Gesture.NONE
        
        # Click debouncing
        self.last_click_time = 0
        self.last_gesture_time = {}
        
        # Gesture history for stability
        self.gesture_history = deque(maxlen=5)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )
    
    def is_finger_extended(self, landmarks, finger_tip_id, finger_pip_id, finger_mcp_id):
        """Check if a finger is extended (straightened)"""
        tip = landmarks[finger_tip_id]
        pip = landmarks[finger_pip_id]
        mcp = landmarks[finger_mcp_id]
        wrist = landmarks[self.WRIST]
        
        # Finger is extended if tip is farther from wrist than PIP
        tip_to_wrist = self.calculate_distance(tip, wrist)
        pip_to_wrist = self.calculate_distance(pip, wrist)
        
        # Also check if tip is higher (lower y value) than MCP
        is_extended = tip_to_wrist > pip_to_wrist and tip.y < mcp.y
        
        return is_extended
    
    def is_thumb_extended(self, landmarks):
        """Special check for thumb extension (horizontal movement)"""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        index_mcp = landmarks[self.INDEX_MCP]
        
        # Thumb extended if tip is far from index finger base
        distance = self.calculate_distance(thumb_tip, index_mcp)
        return distance > self.config.FINGER_EXTENDED_THRESHOLD
    
    def count_extended_fingers(self, landmarks):
        """Count how many fingers are extended"""
        count = 0
        
        # Thumb (special case)
        if self.is_thumb_extended(landmarks):
            count += 1
        
        # Index finger
        if self.is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP):
            count += 1
        
        # Middle finger
        if self.is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP):
            count += 1
        
        # Ring finger
        if self.is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP, self.RING_MCP):
            count += 1
        
        # Pinky
        if self.is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP, self.PINKY_MCP):
            count += 1
        
        return count
    
    def detect_pinch(self, landmarks, finger_tip_id):
        """Detect if finger is pinched with thumb"""
        thumb_tip = landmarks[self.THUMB_TIP]
        finger_tip = landmarks[finger_tip_id]
        
        distance = self.calculate_distance(thumb_tip, finger_tip)
        is_pinched = distance < self.config.PINCH_THRESHOLD
        
        return is_pinched, distance
    
    def detect_gesture(self, landmarks):
        """
        Main gesture detection logic
        Returns: Gesture enum
        """
        # Check for pinches first (most specific)
        index_pinch, index_dist = self.detect_pinch(landmarks, self.INDEX_TIP)
        middle_pinch, middle_dist = self.detect_pinch(landmarks, self.MIDDLE_TIP)
        
        # NEW: Check for thumb-pinky volume control
        thumb_pinky_top_pinch, _ = self.detect_pinch(landmarks, self.PINKY_TIP)
        
        # For bottom pinky, we use the DIP joint (second joint from tip)
        thumb_tip = landmarks[self.THUMB_TIP]
        pinky_dip = landmarks[18]  # PINKY_DIP landmark
        pinky_bottom_dist = self.calculate_distance(thumb_tip, pinky_dip)
        thumb_pinky_bottom_pinch = pinky_bottom_dist < self.config.PINCH_THRESHOLD
        
        # THUMB + PINKY TOP â†’ Volume Up
        if thumb_pinky_top_pinch:
            return Gesture.THUMB_PINKY_TOP
        
        # THUMB + PINKY BOTTOM â†’ Volume Down
        if thumb_pinky_bottom_pinch:
            return Gesture.THUMB_PINKY_BOTTOM
        
        # INDEX + THUMB PINCH â†’ Left Click
        if index_pinch:
            return Gesture.INDEX_THUMB_PINCH
        
        # MIDDLE + THUMB PINCH â†’ Right Click
        if middle_pinch:
            return Gesture.MIDDLE_THUMB_PINCH
        
        # Count extended fingers for other gestures
        extended_count = self.count_extended_fingers(landmarks)
        
        # FIST (0-1 fingers) â†’ Lock cursor
        if extended_count <= 1:
            return Gesture.FIST
        
        # Check for PEACE SIGN (only index and middle extended)
        index_extended = self.is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        middle_extended = self.is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_extended = self.is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP, self.RING_MCP)
        pinky_extended = self.is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP, self.PINKY_MCP)
        
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return Gesture.PEACE_SIGN
        
        # THUMBS UP (only thumb extended, others closed)
        thumb_extended = self.is_thumb_extended(landmarks)
        if thumb_extended and extended_count == 1:
            # Check thumb pointing up
            thumb_tip = landmarks[self.THUMB_TIP]
            wrist = landmarks[self.WRIST]
            if thumb_tip.y < wrist.y:  # Thumb above wrist
                return Gesture.THUMBS_UP
            else:  # Thumb below wrist
                return Gesture.THUMBS_DOWN
        
        # OPEN HAND (4-5 fingers) â†’ Move cursor
        if extended_count >= 4:
            return Gesture.OPEN_HAND
        
        return Gesture.NONE
    
    def update(self, landmarks):
        """
        Update gesture state with debouncing and confirmation
        Returns: (confirmed_gesture, is_new_gesture)
        """
        # Detect current gesture
        raw_gesture = self.detect_gesture(landmarks)
        
        # Add to history
        self.gesture_history.append(raw_gesture)
        
        # Gesture must be consistent across multiple frames
        if len(self.gesture_history) >= self.config.GESTURE_HOLD_FRAMES:
            # Check if last N frames all show same gesture
            recent_gestures = list(self.gesture_history)[-self.config.GESTURE_HOLD_FRAMES:]
            
            if all(g == raw_gesture for g in recent_gestures):
                # Gesture confirmed!
                is_new = (raw_gesture != self.last_confirmed_gesture)
                self.last_confirmed_gesture = raw_gesture
                return raw_gesture, is_new
        
        # Not confirmed yet
        return self.last_confirmed_gesture, False
    
    def can_perform_action(self, gesture_type):
        """Check if enough time has passed since last action (debouncing)"""
        current_time = time.time()
        last_time = self.last_gesture_time.get(gesture_type, 0)
        
        if current_time - last_time > self.config.CLICK_COOLDOWN:
            self.last_gesture_time[gesture_type] = current_time
            return True
        return False

# ==================== MAIN CONTROLLER ====================
class VirtualMouse:
    """Main virtual mouse controller with improved gestures"""
    
    def __init__(self):
        self.config = GestureConfig()
        self.gesture_detector = HandGestureDetector(self.config)
        self.cursor_filter = SmoothingFilter(self.config.CURSOR_SMOOTHING)
        
        # PyAutoGUI setup
        pyautogui.PAUSE = 0.001
        pyautogui.FAILSAFE = True
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # State
        self.scroll_mode = False
        self.cursor_locked = False
        self.last_cursor_pos = None
        self.last_scroll_y = None
        
        # MediaPipe setup
        self.hand_detector = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.drawing = mp.solutions.drawing_utils
        
        # FPS tracking
        self.ptime = 0
        
        print("âœ“ Virtual Mouse initialized")
        print(f"âœ“ Screen: {self.screen_width}x{self.screen_height}")
    
    def calculate_cursor_position(self, landmarks, frame_shape):
        """Calculate smooth cursor position from wrist"""
        frame_height, frame_width = frame_shape[:2]
        
        # Use WRIST for cursor tracking
        wrist = landmarks[self.gesture_detector.WRIST]
        
        # Convert to screen coordinates
        raw_x = self.screen_width / frame_width * (wrist.x * frame_width)
        raw_y = self.screen_height / frame_height * (wrist.y * frame_height)
        
        # Apply smoothing
        smooth_x, smooth_y = self.cursor_filter.update(raw_x, raw_y)
        
        # Apply bounds with margin
        smooth_x = np.clip(smooth_x, self.config.EDGE_MARGIN, 
                          self.screen_width - self.config.EDGE_MARGIN)
        smooth_y = np.clip(smooth_y, self.config.EDGE_MARGIN, 
                          self.screen_height - self.config.EDGE_MARGIN)
        
        return smooth_x, smooth_y
    
    def handle_gesture(self, gesture, is_new, landmarks, frame_shape):
        """Handle detected gesture and perform appropriate action"""
        
        if gesture == Gesture.OPEN_HAND:
            # Move cursor freely
            if not self.cursor_locked:
                cursor_x, cursor_y = self.calculate_cursor_position(landmarks, frame_shape)
                
                # Dead zone check
                if self.last_cursor_pos:
                    last_x, last_y = self.last_cursor_pos
                    distance = np.sqrt((cursor_x - last_x)**2 + (cursor_y - last_y)**2)
                    
                    if distance > self.config.DEAD_ZONE:
                        pyautogui.moveTo(cursor_x, cursor_y, duration=0, _pause=False)
                        self.last_cursor_pos = (cursor_x, cursor_y)
                else:
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0, _pause=False)
                    self.last_cursor_pos = (cursor_x, cursor_y)
            
            return "Moving"
        
        elif gesture == Gesture.INDEX_THUMB_PINCH:
            # Left click
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.click(button='left')
                print("ðŸ–±ï¸ Left Click")
            return "Left Click"
        
        elif gesture == Gesture.MIDDLE_THUMB_PINCH:
            # Right click
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.click(button='right')
                print("ðŸ–±ï¸ Right Click")
            return "Right Click"
        
        elif gesture == Gesture.PEACE_SIGN:
            # Scroll mode
            if is_new:
                self.scroll_mode = not self.scroll_mode
                print(f"ðŸ“œ Scroll: {'ON' if self.scroll_mode else 'OFF'}")
            
            # Perform scrolling if in scroll mode
            if self.scroll_mode:
                index_tip = landmarks[self.gesture_detector.INDEX_TIP]
                current_y = index_tip.y
                
                if self.last_scroll_y is not None:
                    delta_y = current_y - self.last_scroll_y
                    
                    # Scroll (negative delta = up, positive = down)
                    if abs(delta_y) > 0.01:  # Threshold
                        scroll_amount = int(-delta_y * 500)  # Scale factor
                        pyautogui.scroll(scroll_amount)
                
                self.last_scroll_y = current_y
            else:
                self.last_scroll_y = None
            
            return f"Scroll {'ON' if self.scroll_mode else 'OFF'}"
        
        elif gesture == Gesture.FIST:
            # Lock cursor
            if is_new:
                self.cursor_locked = not self.cursor_locked
                print(f"ðŸ”’ {'LOCKED' if self.cursor_locked else 'UNLOCKED'}")
            return f"{'LOCKED' if self.cursor_locked else 'FREE'}"
        
        elif gesture == Gesture.THUMB_PINKY_TOP:
            # Volume up - thumb touches pinky tip
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.press('volumeup')
                print("ðŸ”Š Volume Up")
            return "Vol Up"
        
        elif gesture == Gesture.THUMB_PINKY_BOTTOM:
            # Volume down - thumb touches pinky bottom
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.press('volumedown')
                print("ðŸ”‰ Volume Down")
            return "Vol Down"
        
        return "No hand"
    
    def draw_ui(self, frame, gesture, status_text):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Status box background
        cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, 80), (0, 255, 0), 2)
        
        # Current gesture
        gesture_text = gesture.name.replace('_', ' ')
        cv2.putText(frame, f"{gesture_text}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Status
        cv2.putText(frame, f"{status_text}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Open Hand - Move",
            "Index+Thumb - Left Click",
            "Middle+Thumb - Right Click",
            "Peace - Toggle Scroll",
            "Fist - Lock Cursor",
            "Thumb+Pinky - Volume",
            "",
            "ESC - Exit"
        ]
        
        y_pos = h - 160
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_pos += 18
    
    def run(self):
        """Main loop - PIP mode only"""
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("âŒ Error: Could not open camera")
            return
        
        print("\n" + "="*50)
        print("HAND GESTURE VIRTUAL MOUSE")
        print("="*50)
        print("âœ‹ Open Hand - Move cursor")
        print("ðŸ‘Œ Index+Thumb - Left Click")
        print("ðŸ¤ Middle+Thumb - Right Click")
        print("âœŒï¸  Peace Sign - Toggle Scroll")
        print("âœŠ Fist - Lock/Unlock")
        print("ðŸ¤™ Thumb+Pinky - Volume Control")
        print("\nESC - Exit")
        print("="*50 + "\n")
        
        # Create PIP window
        pip_window = tk.Toplevel()
        pip_window.title("Hand Mouse")
        pip_window.geometry("320x220+10+5")
        pip_window.attributes('-topmost', True)
        pip_window.attributes('-alpha', 0.4)
        pip_window.resizable(False, False)
        pip_window.overrideredirect(True)
        
        # Position at bottom-right
        screen_width = pip_window.winfo_screenwidth()
        screen_height = pip_window.winfo_screenheight()
        pip_window.geometry(f"320x240+{screen_width-340}+{screen_height-280}")
        
        pip_label = tk.Label(pip_window)
        pip_label.pack()
        
        print("ðŸ–¼ï¸  Running in PIP mode")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output = self.hand_detector.process(rgb_frame)
                
                gesture = Gesture.NONE
                status_text = "No hand"
                
                # Process hand landmarks
                if output.multi_hand_landmarks:
                    for hand_landmarks in output.multi_hand_landmarks:
                        # Draw hand skeleton
                        self.drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp.solutions.hands.HAND_CONNECTIONS,
                            self.drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Detect gesture
                        landmarks = hand_landmarks.landmark
                        gesture, is_new = self.gesture_detector.update(landmarks)
                        
                        # Handle gesture
                        status_text = self.handle_gesture(gesture, is_new, landmarks, frame.shape)
                
                # Draw UI
                self.draw_ui(frame, gesture, status_text)
                
                # Update PIP window
                pip_frame = cv2.resize(frame, (320, 240))
                pip_frame_rgb = cv2.cvtColor(pip_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(pip_frame_rgb)
                photo = ImageTk.PhotoImage(image=pil_image)
                pip_label.config(image=photo)
                pip_label.image = photo
                
                try:
                    pip_window.update()
                except tk.TclError:
                    break
                
                # Check for ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        
        except KeyboardInterrupt:
            print("\nâš ï¸  Interrupted")
        except Exception as e:
            print(f"\nâŒ Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if pip_window:
                pip_window.destroy()
            print("âœ“ Cleanup complete")

# ==================== MAIN FUNCTION FOR GUI ====================
def handmouse():
    """Main function to be called from GUI"""
    mouse = VirtualMouse()
    mouse.run()

# ==================== STANDALONE EXECUTION ====================
if __name__ == "__main__":
    handmouse()