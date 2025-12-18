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
    PINCH_THRESHOLD = 0.05
    FINGER_EXTENDED_THRESHOLD = 0.1
    
    # Debounce settings
    CLICK_COOLDOWN = 0.5
    GESTURE_HOLD_FRAMES = 3
    
    # Relative movement settings (NEW)
    USE_RELATIVE_MOVEMENT = True
    HAND_DEAD_ZONE_RADIUS = 0.015  # No movement within this radius
    HAND_CONTROL_RADIUS = 0.12  # Maximum detection radius
    HAND_SPEED_MULTIPLIER = 300.0  # Cursor speed
    HAND_MAX_SPEED = 4000.0  # Maximum pixels per second
    HAND_CENTER_SMOOTHING = 0.2  # How quickly center recalibrates
    
    # Smoothing (for absolute mode)
    CURSOR_SMOOTHING = 7
    
    # Dead zone for cursor stability
    DEAD_ZONE = 15
    
    # Screen edge margin
    EDGE_MARGIN = 50

# ==================== GESTURE TYPES ====================
class Gesture(Enum):
    """All possible gestures"""
    NONE = 0
    OPEN_HAND = 1
    INDEX_THUMB_PINCH = 2
    MIDDLE_THUMB_PINCH = 3
    PEACE_SIGN = 4
    FIST = 5
    THUMB_PINKY_TOP = 8
    THUMB_PINKY_BOTTOM = 9

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

# ==================== RELATIVE HAND CONTROLLER ====================
class HandRelativeController:
    """Controls cursor movement relative to hand position (velocity-based)"""
    
    def __init__(self, config, screen_size):
        self.config = config
        self.screen_w, self.screen_h = screen_size
        
        # Reference position (center of control circle)
        self.reference_hand_pos = None
        self.current_hand_pos = None
        
        # Smoothing for hand position
        self.hand_filter = SmoothingFilter(window_size=3)
        
        # Last update time for velocity calculation
        self.last_update_time = time.time()
        
        print("✓ Relative hand controller initialized")
        print(f"  - Dead zone: {config.HAND_DEAD_ZONE_RADIUS:.3f}")
        print(f"  - Control radius: {config.HAND_CONTROL_RADIUS:.3f}")
        print(f"  - Speed multiplier: {config.HAND_SPEED_MULTIPLIER}")
    
    def get_hand_position(self, wrist_landmark):
        """Extract and smooth hand position from wrist landmark"""
        x = wrist_landmark.x
        y = wrist_landmark.y
        
        # Apply smoothing
        smooth_x, smooth_y = self.hand_filter.update(x, y)
        
        return smooth_x, smooth_y
    
    def update_reference_position(self, wrist_landmark, force=False):
        """
        Update the reference hand position (center of control circle)
        Called when hand is not moving cursor or on initialization
        """
        new_hand_pos = self.get_hand_position(wrist_landmark)
        
        if self.reference_hand_pos is None or force:
            # First time or forced update
            self.reference_hand_pos = new_hand_pos
        else:
            # Smooth update to avoid jumps
            ref_x, ref_y = self.reference_hand_pos
            new_x, new_y = new_hand_pos
            
            alpha = self.config.HAND_CENTER_SMOOTHING
            self.reference_hand_pos = (
                alpha * new_x + (1 - alpha) * ref_x,
                alpha * new_y + (1 - alpha) * ref_y
            )
    
    def calculate_speed_multiplier(self, distance):
        """
        Calculate speed multiplier based on distance from center
        Returns value between 0 and 1
        """
        dead_zone = self.config.HAND_DEAD_ZONE_RADIUS
        max_radius = self.config.HAND_CONTROL_RADIUS
        
        if distance <= dead_zone:
            return 0.0
        
        # Map distance from dead_zone to max_radius -> 0 to 1
        normalized = (distance - dead_zone) / (max_radius - dead_zone)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Exponential curve for smooth acceleration
        return normalized ** 1.5
    
    def calculate_cursor_velocity(self, wrist_landmark):
        """
        Calculate cursor velocity based on hand displacement from reference
        Returns (vx, vy) in pixels per second
        """
        if self.reference_hand_pos is None:
            self.update_reference_position(wrist_landmark, force=True)
            return 0.0, 0.0
        
        # Get current hand position
        self.current_hand_pos = self.get_hand_position(wrist_landmark)
        
        # Calculate displacement vector
        ref_x, ref_y = self.reference_hand_pos
        cur_x, cur_y = self.current_hand_pos
        
        dx = cur_x - ref_x
        dy = cur_y - ref_y
        
        # Calculate distance
        distance = np.sqrt(dx**2 + dy**2)
        
        # Get speed multiplier based on distance
        speed_mult = self.calculate_speed_multiplier(distance)
        
        if speed_mult == 0.0:
            return 0.0, 0.0
        
        # Normalize direction
        if distance > 0:
            direction_x = dx / distance
            direction_y = dy / distance
        else:
            return 0.0, 0.0
        
        # Calculate base speed
        base_speed = speed_mult * self.config.HAND_SPEED_MULTIPLIER
        
        # Apply max speed limit
        speed = min(base_speed, self.config.HAND_MAX_SPEED)
        
        # Calculate velocity components
        vx = direction_x * speed
        vy = direction_y * speed
        
        return vx, vy
    
    def move_cursor(self, wrist_landmark):
        """
        Move cursor based on hand position
        Returns True if cursor was moved, False otherwise
        """
        # Calculate velocity
        vx, vy = self.calculate_cursor_velocity(wrist_landmark)
        
        if abs(vx) < 1 and abs(vy) < 1:
            return False
        
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Clamp dt to avoid huge jumps
        dt = min(dt, 0.1)
        
        # Calculate displacement
        dx = vx * dt
        dy = vy * dt
        
        # Get current cursor position
        current_x, current_y = pyautogui.position()
        
        # Calculate new position
        new_x = current_x + dx
        new_y = current_y + dy
        
        # Apply bounds
        new_x = np.clip(new_x, 0, self.screen_w - 1)
        new_y = np.clip(new_y, 0, self.screen_h - 1)
        
        # Move cursor
        pyautogui.moveTo(new_x, new_y, duration=0, _pause=False)
        
        return True
    
    def get_debug_info(self):
        """Get debug information for visualization"""
        if self.reference_hand_pos is None or self.current_hand_pos is None:
            return {
                'has_reference': False,
                'distance': 0.0,
                'speed_mult': 0.0
            }
        
        ref_x, ref_y = self.reference_hand_pos
        cur_x, cur_y = self.current_hand_pos
        
        dx = cur_x - ref_x
        dy = cur_y - ref_y
        distance = np.sqrt(dx**2 + dy**2)
        
        speed_mult = self.calculate_speed_multiplier(distance)
        
        return {
            'has_reference': True,
            'reference_pos': self.reference_hand_pos,
            'current_pos': self.current_hand_pos,
            'distance': distance,
            'speed_mult': speed_mult,
            'in_dead_zone': distance <= self.config.HAND_DEAD_ZONE_RADIUS,
            'displacement': (dx, dy)
        }
    
    def reset(self):
        """Reset the controller state"""
        self.reference_hand_pos = None
        self.current_hand_pos = None
        self.hand_filter.reset()
        self.last_update_time = time.time()

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
        
        tip_to_wrist = self.calculate_distance(tip, wrist)
        pip_to_wrist = self.calculate_distance(pip, wrist)
        
        is_extended = tip_to_wrist > pip_to_wrist and tip.y < mcp.y
        
        return is_extended
    
    def is_thumb_extended(self, landmarks):
        """Special check for thumb extension (horizontal movement)"""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        index_mcp = landmarks[self.INDEX_MCP]
        
        distance = self.calculate_distance(thumb_tip, index_mcp)
        return distance > self.config.FINGER_EXTENDED_THRESHOLD
    
    def count_extended_fingers(self, landmarks):
        """Count how many fingers are extended"""
        count = 0
        
        if self.is_thumb_extended(landmarks):
            count += 1
        
        if self.is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP):
            count += 1
        
        if self.is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP):
            count += 1
        
        if self.is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP, self.RING_MCP):
            count += 1
        
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
        """Main gesture detection logic"""
        # Check for pinches first
        index_pinch, index_dist = self.detect_pinch(landmarks, self.INDEX_TIP)
        middle_pinch, middle_dist = self.detect_pinch(landmarks, self.MIDDLE_TIP)
        
        # Thumb-pinky volume control
        thumb_pinky_top_pinch, _ = self.detect_pinch(landmarks, self.PINKY_TIP)
        
        thumb_tip = landmarks[self.THUMB_TIP]
        pinky_dip = landmarks[18]
        pinky_bottom_dist = self.calculate_distance(thumb_tip, pinky_dip)
        thumb_pinky_bottom_pinch = pinky_bottom_dist < self.config.PINCH_THRESHOLD
        
        if thumb_pinky_top_pinch:
            return Gesture.THUMB_PINKY_TOP
        
        if thumb_pinky_bottom_pinch:
            return Gesture.THUMB_PINKY_BOTTOM
        
        if index_pinch:
            return Gesture.INDEX_THUMB_PINCH
        
        if middle_pinch:
            return Gesture.MIDDLE_THUMB_PINCH
        
        # Count extended fingers
        extended_count = self.count_extended_fingers(landmarks)
        
        if extended_count <= 1:
            return Gesture.FIST
        
        # Check for PEACE SIGN
        index_extended = self.is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        middle_extended = self.is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_extended = self.is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP, self.RING_MCP)
        pinky_extended = self.is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP, self.PINKY_MCP)
        
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return Gesture.PEACE_SIGN
        
        if extended_count >= 4:
            return Gesture.OPEN_HAND
        
        return Gesture.NONE
    
    def update(self, landmarks):
        """Update gesture state with debouncing"""
        raw_gesture = self.detect_gesture(landmarks)
        
        self.gesture_history.append(raw_gesture)
        
        if len(self.gesture_history) >= self.config.GESTURE_HOLD_FRAMES:
            recent_gestures = list(self.gesture_history)[-self.config.GESTURE_HOLD_FRAMES:]
            
            if all(g == raw_gesture for g in recent_gestures):
                is_new = (raw_gesture != self.last_confirmed_gesture)
                self.last_confirmed_gesture = raw_gesture
                return raw_gesture, is_new
        
        return self.last_confirmed_gesture, False
    
    def can_perform_action(self, gesture_type):
        """Check if enough time has passed since last action"""
        current_time = time.time()
        last_time = self.last_gesture_time.get(gesture_type, 0)
        
        if current_time - last_time > self.config.CLICK_COOLDOWN:
            self.last_gesture_time[gesture_type] = current_time
            return True
        return False

# ==================== MAIN CONTROLLER ====================
class VirtualMouse:
    """Main virtual mouse controller with relative movement"""
    
    def __init__(self):
        self.config = GestureConfig()
        self.gesture_detector = HandGestureDetector(self.config)
        
        # PyAutoGUI setup
        pyautogui.PAUSE = 0.001
        pyautogui.FAILSAFE = False
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize movement controller
        if self.config.USE_RELATIVE_MOVEMENT:
            self.hand_controller = HandRelativeController(
                self.config, 
                (self.screen_width, self.screen_height)
            )
            print("✓ Using RELATIVE hand movement (velocity-based)")
        else:
            self.cursor_filter = SmoothingFilter(self.config.CURSOR_SMOOTHING)
            print("✓ Using ABSOLUTE hand movement (position-based)")
        
        # State
        self.scroll_mode = False
        self.cursor_locked = False
        self.last_scroll_y = None
        
        # MediaPipe setup
        self.hand_detector = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.drawing = mp.solutions.drawing_utils
        
        print("✓ Virtual Mouse initialized")
        print(f"✓ Screen: {self.screen_width}x{self.screen_height}")
    
    def draw_hand_control_circle(self, frame, hand_debug):
        """Draw the hand control circle and current position"""
        if not hand_debug['has_reference']:
            return
        
        h, w = frame.shape[:2]
        
        # Get positions in pixel coordinates
        ref_x, ref_y = hand_debug['reference_pos']
        ref_px = int(ref_x * w)
        ref_py = int(ref_y * h)
        
        cur_x, cur_y = hand_debug['current_pos']
        cur_px = int(cur_x * w)
        cur_py = int(cur_y * h)
        
        # Draw dead zone circle (inner - no movement)
        dead_zone_radius = int(self.config.HAND_DEAD_ZONE_RADIUS * w)
        cv2.circle(frame, (ref_px, ref_py), dead_zone_radius, (100, 100, 100), 1)
        
        # Draw control circle (outer)
        control_radius = int(self.config.HAND_CONTROL_RADIUS * w)
        color = (0, 255, 255) if not self.cursor_locked else (100, 100, 100)
        cv2.circle(frame, (ref_px, ref_py), control_radius, color, 2)
        
        # Draw center point
        cv2.circle(frame, (ref_px, ref_py), 4, (0, 255, 0), -1)
        
        # Draw current hand position
        hand_color = (0, 255, 255) if not hand_debug['in_dead_zone'] else (100, 100, 100)
        cv2.circle(frame, (cur_px, cur_py), 6, hand_color, -1)
        
        # Draw displacement vector
        if not hand_debug['in_dead_zone'] and not self.cursor_locked:
            cv2.arrowedLine(frame, (ref_px, ref_py), (cur_px, cur_py), 
                          (0, 255, 255), 2, tipLength=0.3)
        
        # Draw speed indicator bar
        speed_mult = hand_debug['speed_mult']
        bar_height = 100
        bar_width = 20
        bar_x = w - 40
        bar_y = 50
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Speed fill
        fill_height = int(bar_height * speed_mult)
        if fill_height > 0:
            cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (0, 255, 255), -1)
        
        # Speed percentage
        cv2.putText(frame, f"{int(speed_mult * 100)}%", 
                   (bar_x - 30, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def handle_gesture(self, gesture, is_new, landmarks, frame_shape):
        """Handle detected gesture and perform appropriate action"""
        
        if gesture == Gesture.OPEN_HAND:
            # Move cursor with relative movement
            if not self.cursor_locked:
                if self.config.USE_RELATIVE_MOVEMENT:
                    wrist = landmarks[self.gesture_detector.WRIST]
                    self.hand_controller.move_cursor(wrist)
                else:
                    # Old absolute positioning (fallback)
                    cursor_x, cursor_y = self.calculate_cursor_position(landmarks, frame_shape)
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0, _pause=False)
            
            return "Moving"
        
        elif gesture == Gesture.INDEX_THUMB_PINCH:
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.click(button='left')
                print("🖱️ Left Click")
            return "Left Click"
        
        elif gesture == Gesture.MIDDLE_THUMB_PINCH:
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.click(button='right')
                print("🖱️ Right Click")
            return "Right Click"
        
        elif gesture == Gesture.PEACE_SIGN:
            if is_new:
                self.scroll_mode = not self.scroll_mode
                print(f"📜 Scroll: {'ON' if self.scroll_mode else 'OFF'}")
            
            if self.scroll_mode:
                index_tip = landmarks[self.gesture_detector.INDEX_TIP]
                current_y = index_tip.y
                
                if self.last_scroll_y is not None:
                    delta_y = current_y - self.last_scroll_y
                    
                    if abs(delta_y) > 0.01:
                        scroll_amount = int(-delta_y * 500)
                        pyautogui.scroll(scroll_amount)
                
                self.last_scroll_y = current_y
            else:
                self.last_scroll_y = None
            
            return f"Scroll {'ON' if self.scroll_mode else 'OFF'}"
        
        elif gesture == Gesture.FIST:
            if is_new:
                self.cursor_locked = not self.cursor_locked
                if self.config.USE_RELATIVE_MOVEMENT:
                    if not self.cursor_locked:
                        # When unlocking, recalibrate center at current hand position
                        wrist = landmarks[self.gesture_detector.WRIST]
                        self.hand_controller.update_reference_position(wrist, force=True)
                        print("🔓 UNLOCKED - Center recalibrated")
                    else:
                        print("🔒 LOCKED")
                else:
                    print(f"🔒 {'LOCKED' if self.cursor_locked else 'UNLOCKED'}")
            return f"{'LOCKED' if self.cursor_locked else 'FREE'}"
        
        elif gesture == Gesture.THUMB_PINKY_TOP:
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                pyautogui.press('volumeup')
                print("🔊 Volume Up")
            return "Vol Up"
        
        elif gesture == Gesture.THUMB_PINKY_BOTTOM:
            if is_new and self.gesture_detector.can_perform_action(gesture):
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                pyautogui.press('volumedown')
                print("🔉 Volume Down")
            return "Vol Down"
        
        # Update reference position when hand is idle (not open hand)
        if self.config.USE_RELATIVE_MOVEMENT and gesture != Gesture.OPEN_HAND:
            wrist = landmarks[self.gesture_detector.WRIST]
            self.hand_controller.update_reference_position(wrist)
        
        return "No hand"
    
    def draw_ui(self, frame, gesture, status_text, hand_debug=None):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Draw hand control circle if in relative mode
        if self.config.USE_RELATIVE_MOVEMENT and hand_debug:
            self.draw_hand_control_circle(frame, hand_debug)
        
        # Status box background
        cv2.rectangle(frame, (10, 10), (320, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 110), (0, 255, 0), 2)
        
        # Mode indicator
        mode_text = "RELATIVE" if self.config.USE_RELATIVE_MOVEMENT else "ABSOLUTE"
        cv2.putText(frame, f"Mode: {mode_text}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Current gesture
        gesture_text = gesture.name.replace('_', ' ')
        cv2.putText(frame, f"{gesture_text}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Status
        cv2.putText(frame, f"{status_text}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        if self.config.USE_RELATIVE_MOVEMENT:
            instructions = [
                "RELATIVE MODE - Move hand from center",
                "Open Hand - Move cursor (velocity)",
                "Index+Thumb - Left Click",
                "Middle+Thumb - Right Click",
                "Peace - Toggle Scroll",
                "Fist - Lock/Unlock & Recalibrate",
                "Thumb+Pinky - Volume",
                "",
                "ESC - Exit"
            ]
        else:
            instructions = [
                "ABSOLUTE MODE - Direct positioning",
                "Open Hand - Move cursor",
                "Index+Thumb - Left Click",
                "Middle+Thumb - Right Click",
                "Peace - Toggle Scroll",
                "Fist - Lock Cursor",
                "Thumb+Pinky - Volume",
                "",
                "ESC - Exit"
            ]
        
        y_pos = h - 180
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
            print("❌ Error: Could not open camera")
            return
        
        print("\n" + "="*50)
        print("HAND GESTURE VIRTUAL MOUSE - RELATIVE MODE")
        print("="*50)
        if self.config.USE_RELATIVE_MOVEMENT:
            print("🎯 RELATIVE Movement (Velocity-based)")
            print("   - Hand stays in comfortable position")
            print("   - Move from center for cursor control")
            print("   - Fist unlocks & recalibrates center")
        else:
            print("🎯 ABSOLUTE Movement (Position-based)")
        print("✋ Open Hand - Move cursor")
        print("👆 Index+Thumb - Left Click")
        print("🤘 Middle+Thumb - Right Click")
        print("✌️  Peace Sign - Toggle Scroll")
        print("✊ Fist - Lock/Unlock")
        print("🤙 Thumb+Pinky - Volume Control")
        print("\nESC - Exit")
        print("="*50 + "\n")
        
        # Create PIP window with proper Tk root
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        pip_window = tk.Toplevel(root)
        pip_window.title("Hand Mouse")
        pip_window.geometry("320x240+1580+780")
        pip_window.attributes('-topmost', True)
        pip_window.attributes('-alpha', 0.5)
        pip_window.resizable(False, False)
        pip_window.overrideredirect(True)
        
        pip_label = tk.Label(pip_window)
        pip_label.pack()
        
        print("📼️ Running in PIP mode")
        
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
                hand_debug = None
                
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
                        
                        # Get debug info for visualization
                        if self.config.USE_RELATIVE_MOVEMENT:
                            hand_debug = self.hand_controller.get_debug_info()
                
                # Draw UI
                self.draw_ui(frame, gesture, status_text, hand_debug)
                
                # Update PIP window
                try:
                    pip_frame = cv2.resize(frame, (320, 240))
                    pip_frame_rgb = cv2.cvtColor(pip_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(pip_frame_rgb)
                    photo = ImageTk.PhotoImage(image=pil_image, master=pip_window)
                    
                    pip_label.config(image=photo)
                    pip_label.image = photo  # Keep reference
                    
                    pip_window.update_idletasks()
                    pip_window.update()
                except tk.TclError as e:
                    print(f"PIP window closed: {e}")
                    break
                except Exception as e:
                    print(f"PIP update error: {e}")
                    # Continue even if PIP fails
                    pass
                
                # Check for ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        
        except KeyboardInterrupt:
            print("\n⛔️ Interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            try:
                if pip_window:
                    pip_window.destroy()
                if root:
                    root.destroy()
            except:
                pass
            print("✓ Cleanup complete")

# ==================== MAIN FUNCTION FOR GUI ====================
def handmouse():
    """Main function to be called from GUI"""
    mouse = VirtualMouse()
    mouse.run()

# ==================== STANDALONE EXECUTION ====================
if __name__ == "__main__":
    handmouse()