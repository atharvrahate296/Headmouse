"""
Enhanced Virtual Mouse Control System - Mouth-Controlled with Relative Nose Movement
Uses MediaPipe Face Mesh for robust face tracking with mouth-open detection
Cursor moves relative to nose position (velocity-based) instead of absolute positioning
"""

import sys
import time
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Deque
import numpy as np

# Handle imports with proper error messaging
try:
    import cv2
    import mediapipe as mp
    import pyautogui as pag
except ImportError as e:
    print(f"❌ FATAL ERROR: Missing Required Library")
    print(f"Details: {e}")
    print("\nPlease install required libraries:")
    print("pip install opencv-python mediapipe pyautogui")
    sys.exit(1)


@dataclass
class Config:
    """Configuration for virtual mouse behavior - CALIBRATED VALUES"""
    
    
    # CALIBRATED THRESHOLDS - Based on your facial measurements
    
    
    # Eye detection thresholds
    eye_closure_thresh: float = 0.265
    blink_consecutive_frames: int = 2
    wink_ar_diff_thresh: float = 0.045
    wink_consecutive_frames: int = 6
    
    # Mouth detection thresholds
    mouth_open_thresh: float = 0.500
    mouth_consecutive_frames: int = 5
    
    
    # RELATIVE NOSE MOVEMENT SETTINGS (NEW)
    
    use_relative_movement: bool = True  # Toggle between relative/absolute
    nose_dead_zone_radius: float = 0.005  # No movement within this radius (normalized)
    nose_control_radius: float = 0.06  # Maximum detection radius (normalized)
    nose_speed_multiplier: float = 100.0  # Cursor pixels per second per unit displacement
    nose_speed_curve: str = "exponential"  # "linear", "exponential", "squared"
    nose_max_speed: float = 20000.0  # Maximum cursor speed (pixels per second)
    nose_center_smoothing: float = 0.3  # Smoothing for center position updates (0-1)
    
    
    # Mouse control (for absolute mode)
    
    cursor_smoothing: int = 5
    movement_multiplier: float = 1.8
    dead_zone: float = 0.02
    edge_margin: int = 50
    
    # Camera settings
    cam_width: int = 640
    cam_height: int = 480
    
    # Visual feedback
    show_face_mesh: bool = True
    show_eye_markers: bool = True
    show_mouth_markers: bool = True
    show_nose_circle: bool = True  # NEW: Show nose control circle
    show_stats: bool = True
    show_debug_values: bool = False
    
    def save(self, path: str = "mouse_config.json"):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = "mouse_config.json"):
        """Load configuration from file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            import inspect
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}
            
            print(f"✓ Loaded calibrated config from {path}")
            return cls(**filtered_data)
        except FileNotFoundError:
            print(f"⚠️  Config file not found, using defaults")
            return cls()
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            print("   Using default values instead")
            return cls()


class SmoothingFilter:
    """Exponential moving average filter for smooth cursor movement"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.alpha = 2 / (window_size + 1)
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Add new point and return smoothed coordinates"""
        self.history.append((x, y))
        
        if len(self.history) == 1:
            return x, y
        
        # Exponential moving average
        smooth_x = x
        smooth_y = y
        
        if len(self.history) > 1:
            prev_x, prev_y = self.history[-2]
            smooth_x = self.alpha * x + (1 - self.alpha) * prev_x
            smooth_y = self.alpha * y + (1 - self.alpha) * prev_y
        
        return smooth_x, smooth_y
    
    def reset(self):
        """Clear history"""
        self.history.clear()


class NoseRelativeController:
    """Controls cursor movement relative to nose position (velocity-based)"""
    
    # Nose tip landmark
    NOSE_TIP = 1  # Main nose tip landmark
    NOSE_BRIDGE = 4  # Alternative: nose bridge
    
    def __init__(self, config: Config, screen_size: Tuple[int, int]):
        self.config = config
        self.screen_w, self.screen_h = screen_size
        
        # Reference position (center of control circle)
        self.reference_nose_pos: Optional[Tuple[float, float]] = None
        self.current_nose_pos: Optional[Tuple[float, float]] = None
        
        # Smoothing for nose position
        self.nose_filter = SmoothingFilter(window_size=3)
        
        # Smoothing for reference position updates
        self.reference_smoothing_factor = config.nose_center_smoothing
        
        # Last update time for velocity calculation
        self.last_update_time = time.time()
        
        print(f"✓ Relative nose controller initialized")
        print(f"  - Dead zone: {config.nose_dead_zone_radius:.3f}")
        print(f"  - Control radius: {config.nose_control_radius:.3f}")
        print(f"  - Speed curve: {config.nose_speed_curve}")
        print(f"  - Max speed: {config.nose_max_speed} px/s")
    
    def get_nose_position(self, landmarks) -> Tuple[float, float]:
        """Extract and smooth nose position from landmarks"""
        nose_landmark = landmarks[self.NOSE_TIP]
        
        # Use normalized coordinates (0-1 range)
        x = nose_landmark.x
        y = nose_landmark.y
        
        # Apply smoothing
        smooth_x, smooth_y = self.nose_filter.update(x, y)
        
        return smooth_x, smooth_y
    
    def update_reference_position(self, landmarks, force: bool = False):
        """
        Update the reference nose position (center of control circle)
        Called when mouth is closed or on initialization
        """
        new_nose_pos = self.get_nose_position(landmarks)
        
        if self.reference_nose_pos is None or force:
            # First time or forced update
            self.reference_nose_pos = new_nose_pos
        else:
            # Smooth update to avoid jumps
            ref_x, ref_y = self.reference_nose_pos
            new_x, new_y = new_nose_pos
            
            alpha = self.reference_smoothing_factor
            self.reference_nose_pos = (
                alpha * new_x + (1 - alpha) * ref_x,
                alpha * new_y + (1 - alpha) * ref_y
            )
    
    def calculate_speed_multiplier(self, distance: float) -> float:
        """
        Calculate speed multiplier based on distance from center
        Returns value between 0 and 1
        """
        # Normalize distance to 0-1 range
        dead_zone = self.config.nose_dead_zone_radius
        max_radius = self.config.nose_control_radius
        
        if distance <= dead_zone:
            return 0.0
        
        # Map distance from dead_zone to max_radius -> 0 to 1
        normalized = (distance - dead_zone) / (max_radius - dead_zone)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Apply speed curve
        if self.config.nose_speed_curve == "linear":
            return normalized
        elif self.config.nose_speed_curve == "exponential":
            return normalized ** 1.5  # Gentle exponential
        elif self.config.nose_speed_curve == "squared":
            return normalized ** 2
        else:
            return normalized
    
    def calculate_cursor_velocity(self, landmarks) -> Tuple[float, float]:
        """
        Calculate cursor velocity based on nose displacement from reference
        Returns (vx, vy) in pixels per second
        """
        if self.reference_nose_pos is None:
            self.update_reference_position(landmarks, force=True)
            return 0.0, 0.0
        
        # Get current nose position
        self.current_nose_pos = self.get_nose_position(landmarks)
        
        # Calculate displacement vector
        ref_x, ref_y = self.reference_nose_pos
        cur_x, cur_y = self.current_nose_pos
        
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
        base_speed = speed_mult * self.config.nose_speed_multiplier
        
        # Apply max speed limit
        speed = min(base_speed, self.config.nose_max_speed)
        
        # Calculate velocity components (convert to screen pixels)
        vx = direction_x * speed
        vy = direction_y * speed
        
        return vx, vy
    
    def move_cursor(self, landmarks) -> bool:
        """
        Move cursor based on nose position
        Returns True if cursor was moved, False otherwise
        """
        # Calculate velocity
        vx, vy = self.calculate_cursor_velocity(landmarks)
        
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
        current_x, current_y = pag.position()
        
        # Calculate new position
        new_x = current_x + dx
        new_y = current_y + dy
        
        # Apply bounds
        new_x = np.clip(new_x, 0, self.screen_w - 1)
        new_y = np.clip(new_y, 0, self.screen_h - 1)
        
        # Move cursor
        pag.moveTo(new_x, new_y, duration=0, _pause=False)
        
        return True
    
    def get_debug_info(self) -> dict:
        """Get debug information for visualization"""
        if self.reference_nose_pos is None or self.current_nose_pos is None:
            return {
                'has_reference': False,
                'distance': 0.0,
                'speed_mult': 0.0,
                'velocity': (0.0, 0.0)
            }
        
        ref_x, ref_y = self.reference_nose_pos
        cur_x, cur_y = self.current_nose_pos
        
        dx = cur_x - ref_x
        dy = cur_y - ref_y
        distance = np.sqrt(dx**2 + dy**2)
        
        speed_mult = self.calculate_speed_multiplier(distance)
        
        return {
            'has_reference': True,
            'reference_pos': self.reference_nose_pos,
            'current_pos': self.current_nose_pos,
            'distance': distance,
            'speed_mult': speed_mult,
            'in_dead_zone': distance <= self.config.nose_dead_zone_radius,
            'displacement': (dx, dy)
        }
    
    def reset(self):
        """Reset the controller state"""
        self.reference_nose_pos = None
        self.current_nose_pos = None
        self.nose_filter.reset()
        self.last_update_time = time.time()


class MouthDetector:
    """Detects mouth open/closed state for cursor movement control"""
    
    # Mouth landmarks (outer and inner lips)
    MOUTH_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    MOUTH_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    MOUTH_LEFT = [61, 146]
    MOUTH_RIGHT = [291, 375]
    
    def __init__(self, config: Config):
        self.config = config
        self.mouth_open_counter = 0
        self.mouth_closed_counter = 0
        self.is_mouth_open = False
        self.mouth_ratio_history = deque(maxlen=3)
    
    def calculate_mouth_aspect_ratio(self, landmarks) -> float:
        """Calculate Mouth Aspect Ratio (MAR)"""
        # Get mouth landmark points
        upper_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in self.MOUTH_UPPER])
        lower_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in self.MOUTH_LOWER])
        
        # Calculate vertical distances (multiple measurements)
        vertical_dists = []
        for i in range(min(len(upper_points), len(lower_points))):
            dist = np.linalg.norm(upper_points[i] - lower_points[i])
            vertical_dists.append(dist)
        
        avg_vertical = np.mean(vertical_dists)
        
        # Calculate horizontal distance
        left_point = np.array([landmarks[self.MOUTH_LEFT[0]].x, 
                              landmarks[self.MOUTH_LEFT[0]].y])
        right_point = np.array([landmarks[self.MOUTH_RIGHT[0]].x, 
                               landmarks[self.MOUTH_RIGHT[0]].y])
        horizontal = np.linalg.norm(right_point - left_point)
        
        # MAR = vertical / horizontal
        mar = avg_vertical / (horizontal + 1e-6)
        
        return mar
    
    def detect_mouth_open(self, landmarks) -> bool:
        """
        Detect if mouth is open
        Returns True if mouth is open, False if closed
        """
        mar = self.calculate_mouth_aspect_ratio(landmarks)
        
        # Add to history for smoothing
        self.mouth_ratio_history.append(mar)
        
        # Average over recent history
        if len(self.mouth_ratio_history) > 1:
            mar = np.mean(self.mouth_ratio_history)
        
        # State machine for mouth detection
        if mar > self.config.mouth_open_thresh: 
            self.mouth_open_counter += 1
            self.mouth_closed_counter = 0
            
            if self.mouth_open_counter >= self.config.mouth_consecutive_frames:
                self.is_mouth_open = True
        else:
            self.mouth_closed_counter += 1
            self.mouth_open_counter = 0
            
            if self.mouth_closed_counter >= self.config.mouth_consecutive_frames:
                self.is_mouth_open = False
        
        return self.is_mouth_open
    
    def get_debug_values(self, landmarks) -> dict:
        """Get current mouth detection values for debugging"""
        mar = self.calculate_mouth_aspect_ratio(landmarks)
        
        return {
            'mouth_ratio': mar,
            'is_open': self.is_mouth_open,
            'threshold': self.config.mouth_open_thresh
        }


class EnhancedGestureDetector:
    """
    Advanced gesture detector using multiple landmarks around eyes
    Specifically designed for low-quality camera feeds
    """
    
    # Comprehensive eye region landmarks
    # Left eye with surrounding landmarks
    LEFT_EYE_CORE = [362, 385, 387, 263, 373, 380]
    LEFT_EYE_OUTER = [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    LEFT_EYEBROW = [276, 283, 282, 295, 285]
    LEFT_EYELID_UPPER = [384, 398, 362]
    LEFT_EYELID_LOWER = [381, 380, 374]
    
    # Right eye with surrounding landmarks
    RIGHT_EYE_CORE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_OUTER = [466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
    RIGHT_EYEBROW = [46, 53, 52, 65, 55]
    RIGHT_EYELID_UPPER = [160, 159, 158]
    RIGHT_EYELID_LOWER = [144, 145, 153]
    
    def __init__(self, config: Config):
        self.config = config
        self.blink_counter = 0
        self.wink_counter = 0
        self.last_blink_time = 0
        self.last_click_time = 0
        
        # History for temporal smoothing
        self.left_closure_history = deque(maxlen=3)
        self.right_closure_history = deque(maxlen=3)
    
    def calculate_enhanced_ear(self, eye_core_landmarks, landmarks) -> float:
        """Calculate enhanced Eye Aspect Ratio using core landmarks"""
        points = np.array([(landmarks[i].x, landmarks[i].y, landmarks[i].z) 
                          for i in eye_core_landmarks])
        
        # Multiple vertical measurements
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        
        # Horizontal measurement
        h = np.linalg.norm(points[0] - points[3])
        
        # Standard EAR
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def calculate_eyelid_distance(self, upper_landmarks, lower_landmarks, landmarks) -> float:
        """Calculate vertical distance between eyelids"""
        upper_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in upper_landmarks])
        lower_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in lower_landmarks])
        
        # Average vertical distance
        upper_center = np.mean(upper_points, axis=0)
        lower_center = np.mean(lower_points, axis=0)
        
        distance = np.linalg.norm(upper_center - lower_center)
        return distance
    
    def calculate_eyebrow_eye_ratio(self, eyebrow_landmarks, eye_landmarks, landmarks) -> float:
        """Calculate ratio between eyebrow and eye positions"""
        eyebrow_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                   for i in eyebrow_landmarks])
        eye_points = np.array([(landmarks[i].x, landmarks[i].y) 
                               for i in eye_landmarks])
        
        eyebrow_y = np.mean(eyebrow_points[:, 1])
        eye_y = np.mean(eye_points[:, 1])
        
        # When eye closes, this ratio decreases
        ratio = abs(eyebrow_y - eye_y)
        return ratio
    
    def calculate_comprehensive_eye_closure(self, eye_core, eyelid_upper, 
                                           eyelid_lower, eyebrow, landmarks) -> float:
        """
        Calculate comprehensive eye closure metric combining multiple signals
        Returns value between 0 (open) and 1 (closed)
        """
        # 1. Standard EAR (inverted so higher = more closed)
        ear = self.calculate_enhanced_ear(eye_core, landmarks)
        ear_score = 1.0 - (ear * 15)  # Scale and invert
        ear_score = np.clip(ear_score, 0, 1)
        
        # 2. Eyelid distance (inverted)
        eyelid_dist = self.calculate_eyelid_distance(eyelid_upper, eyelid_lower, landmarks)
        eyelid_score = 1.0 - (eyelid_dist * 30)  # Scale and invert
        eyelid_score = np.clip(eyelid_score, 0, 1)
        
        # 3. Eyebrow-eye ratio (when eye closes, eyebrow stays relatively same)
        eyebrow_ratio = self.calculate_eyebrow_eye_ratio(eyebrow, eye_core, landmarks)
        eyebrow_score = 1.0 - (eyebrow_ratio * 10)
        eyebrow_score = np.clip(eyebrow_score, 0, 1)
        
        # Weighted combination (EAR is most reliable, others are supporting)
        combined = (ear_score * 0.5 + eyelid_score * 0.35 + eyebrow_score * 0.15)
        
        return combined
    
    def detect_blink(self, landmarks) -> bool:
        """
        Enhanced blink detection using comprehensive eye closure metrics
        Works better with low-quality cameras
        """
        # Calculate comprehensive closure for both eyes
        left_closure = self.calculate_comprehensive_eye_closure(
            self.LEFT_EYE_CORE, self.LEFT_EYELID_UPPER, 
            self.LEFT_EYELID_LOWER, self.LEFT_EYEBROW, landmarks
        )
        
        right_closure = self.calculate_comprehensive_eye_closure(
            self.RIGHT_EYE_CORE, self.RIGHT_EYELID_UPPER,
            self.RIGHT_EYELID_LOWER, self.RIGHT_EYEBROW, landmarks
        )
        
        # Add to history for temporal smoothing
        self.left_closure_history.append(left_closure)
        self.right_closure_history.append(right_closure)
        
        # Average over recent history to reduce noise
        if len(self.left_closure_history) > 1:
            left_closure = np.mean(self.left_closure_history)
            right_closure = np.mean(self.right_closure_history)
        
        # BOTH eyes must be closed AND relatively equal (not a wink)
        avg_closure = (left_closure + right_closure) / 2.0
        closure_symmetry = abs(left_closure - right_closure)
        
        # For blink: both eyes closed AND symmetric (difference < 0.06)
        is_blink_state = (avg_closure > self.config.eye_closure_thresh and 
                         closure_symmetry < 0.06)
        
        if is_blink_state:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.config.blink_consecutive_frames:
                # Debounce - prevent multiple detections
                current_time = time.time()
                if current_time - self.last_blink_time > 0.5:
                    self.blink_counter = 0
                    self.last_blink_time = current_time
                    return True
            self.blink_counter = 0
        
        return False
    
    def detect_wink(self, landmarks) -> Optional[str]:
        """
        Enhanced wink detection using comprehensive metrics
        """
        left_closure = self.calculate_comprehensive_eye_closure(
            self.LEFT_EYE_CORE, self.LEFT_EYELID_UPPER,
            self.LEFT_EYELID_LOWER, self.LEFT_EYEBROW, landmarks
        )
        
        right_closure = self.calculate_comprehensive_eye_closure(
            self.RIGHT_EYE_CORE, self.RIGHT_EYELID_UPPER,
            self.RIGHT_EYELID_LOWER, self.RIGHT_EYEBROW, landmarks
        )
        
        # Smooth with history
        self.left_closure_history.append(left_closure)
        self.right_closure_history.append(right_closure)
        
        if len(self.left_closure_history) > 1:
            left_closure = np.mean(self.left_closure_history)
            right_closure = np.mean(self.right_closure_history)
        
        closure_diff = abs(left_closure - right_closure)
        
        # One eye should be significantly more closed than the other
        if closure_diff > self.config.wink_ar_diff_thresh:
            if left_closure > self.config.eye_closure_thresh and left_closure > right_closure:
                wink_type = 'left'
            elif right_closure > self.config.eye_closure_thresh and right_closure > left_closure:
                wink_type = 'right'
            else:
                self.wink_counter = 0
                return None
            
            self.wink_counter += 1
            
            if self.wink_counter >= self.config.wink_consecutive_frames:
                current_time = time.time()
                if current_time - self.last_click_time > 0.5:
                    self.wink_counter = 0
                    self.last_click_time = current_time
                    return wink_type
        else:
            self.wink_counter = 0
        
        return None
    
    def get_debug_values(self, landmarks) -> dict:
        """Get current detection values for debugging/calibration"""
        left_closure = self.calculate_comprehensive_eye_closure(
            self.LEFT_EYE_CORE, self.LEFT_EYELID_UPPER,
            self.LEFT_EYELID_LOWER, self.LEFT_EYEBROW, landmarks
        )
        
        right_closure = self.calculate_comprehensive_eye_closure(
            self.RIGHT_EYE_CORE, self.RIGHT_EYELID_UPPER,
            self.RIGHT_EYELID_LOWER, self.RIGHT_EYEBROW, landmarks
        )
        
        return {
            'left_closure': left_closure,
            'right_closure': right_closure,
            'avg_closure': (left_closure + right_closure) / 2.0,
            'closure_diff': abs(left_closure - right_closure)
        }


class VirtualMouse:
    """Main virtual mouse controller - Mouth-controlled with relative nose movement"""
    
    IRIS_LEFT = [474, 475, 476, 477]
    IRIS_RIGHT = [469, 470, 471, 472]
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        
        # Initialize PyAutoGUI safety
        pag.FAILSAFE = True
        pag.PAUSE = 0.001
        
        # Get screen dimensions
        self.screen_w, self.screen_h = pag.size()
        
        # Initialize components
        self.gesture_detector = EnhancedGestureDetector(self.config)
        self.mouth_detector = MouthDetector(self.config)
        
        # Initialize movement controller based on mode
        if self.config.use_relative_movement:
            self.nose_controller = NoseRelativeController(self.config, (self.screen_w, self.screen_h))
            print("✓ Using RELATIVE nose movement (velocity-based)")
        else:
            self.cursor_filter = SmoothingFilter(self.config.cursor_smoothing)
            print("✓ Using ABSOLUTE iris movement (position-based)")
        
        # State variables
        self.scroll_mode = False
        self.is_moving = False
        self.last_cursor_pos = None
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        print("✓ Mouth-Controlled Virtual Mouse initialized")
        print(f"✓ Screen resolution: {self.screen_w}x{self.screen_h}")
        print(f"✓ Movement mode: {'RELATIVE (nose)' if self.config.use_relative_movement else 'ABSOLUTE (iris)'}")
    
    def calculate_cursor_position_absolute(self, landmarks, frame_shape) -> Tuple[float, float]:
        """Calculate cursor position from iris landmarks (ABSOLUTE MODE)"""
        # Use left iris center (landmark 475)
        iris_landmark = landmarks[475]
        
        # Convert to screen coordinates
        x = iris_landmark.x * self.screen_w
        y = iris_landmark.y * self.screen_h
        
        # Apply smoothing
        smooth_x, smooth_y = self.cursor_filter.update(x, y)
        
        # Apply bounds with edge margin
        smooth_x = np.clip(smooth_x, self.config.edge_margin, 
                          self.screen_w - self.config.edge_margin)
        smooth_y = np.clip(smooth_y, self.config.edge_margin, 
                          self.screen_h - self.config.edge_margin)
        
        return smooth_x, smooth_y
    
    def draw_nose_control_circle(self, frame, nose_debug: dict):
        """Draw the nose control circle and current position"""
        if not nose_debug['has_reference']:
            return
        
        h, w = frame.shape[:2]
        
        # Get positions in pixel coordinates
        ref_x, ref_y = nose_debug['reference_pos']
        ref_px = int(ref_x * w)
        ref_py = int(ref_y * h)
        
        cur_x, cur_y = nose_debug['current_pos']
        cur_px = int(cur_x * w)
        cur_py = int(cur_y * h)
        
        # Draw dead zone circle (inner - no movement)
        dead_zone_radius = int(self.config.nose_dead_zone_radius * w)
        cv2.circle(frame, (ref_px, ref_py), dead_zone_radius, (100, 100, 100), 1)
        
        # Draw control circle (outer - maximum detection)
        control_radius = int(self.config.nose_control_radius * w)
        color = (0, 255, 255) if self.is_moving else (100, 100, 100)
        cv2.circle(frame, (ref_px, ref_py), control_radius, color, 2)
        
        # Draw center point
        cv2.circle(frame, (ref_px, ref_py), 4, (0, 255, 0), -1)
        
        # Draw current nose position
        nose_color = (0, 255, 255) if not nose_debug['in_dead_zone'] else (100, 100, 100)
        cv2.circle(frame, (cur_px, cur_py), 6, nose_color, -1)
        
        # Draw displacement vector
        if not nose_debug['in_dead_zone']:
            cv2.arrowedLine(frame, (ref_px, ref_py), (cur_px, cur_py), 
                          (0, 255, 255), 2, tipLength=0.3)
        
        # Draw speed indicator (bar on the side)
        speed_mult = nose_debug['speed_mult']
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
        
        # Speed percentage text
        cv2.putText(frame, f"{int(speed_mult * 100)}%", 
                   (bar_x - 30, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def draw_visualizations(self, frame, face_landmarks, eye_debug_values: dict, 
                           mouth_debug_values: dict, nose_debug: Optional[dict] = None):
        """Draw visual feedback on frame"""
        h, w = frame.shape[:2]
        
        if self.config.show_face_mesh:
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        
        if self.config.show_eye_markers:
            # Draw iris points
            for idx in self.IRIS_LEFT + self.IRIS_RIGHT:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        if self.config.show_mouth_markers:
            # Draw mouth landmarks
            for idx in self.mouth_detector.MOUTH_UPPER + self.mouth_detector.MOUTH_LOWER:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                color = (0, 255, 255) if self.is_moving else (255, 0, 0)
                cv2.circle(frame, (x, y), 2, color, -1)
        
        # Draw nose control circle (if in relative mode)
        if self.config.use_relative_movement and self.config.show_nose_circle and nose_debug:
            self.draw_nose_control_circle(frame, nose_debug)
        
        if self.config.show_stats:
            y_offset = 30
            
            # Movement mode indicator
            mode_text = "RELATIVE" if self.config.use_relative_movement else "ABSOLUTE"
            cv2.putText(frame, f"MODE: {mode_text}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            
            # Movement status indicator
            if self.is_moving:
                cv2.putText(frame, "MOVING: ACTIVE", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "CURSOR: LOCKED", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            # Mouth status
            mouth_status = "OPEN" if mouth_debug_values['is_open'] else "CLOSED"
            color = (0, 255, 255) if mouth_debug_values['is_open'] else (255, 255, 255)
            cv2.putText(frame, f"MOUTH: {mouth_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
            
            if self.scroll_mode:
                cv2.putText(frame, "SCROLL MODE: ON", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30
            
            # FPS
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            # Nose debug info (if relative mode)
            if self.config.use_relative_movement and nose_debug and nose_debug['has_reference']:
                cv2.putText(frame, f"Distance: {nose_debug['distance']:.4f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 255), 1)
                y_offset += 20
            
            # Debug values for calibration
            if self.config.show_debug_values:
                # Eye debug
                cv2.putText(frame, 
                           f"Eye L: {eye_debug_values['left_closure']:.3f} | "
                           f"R: {eye_debug_values['right_closure']:.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 0, 255), 1)
                y_offset += 25
                
                # Mouth debug
                cv2.putText(frame, 
                           f"Mouth: {mouth_debug_values['mouth_ratio']:.3f} | "
                           f"Thresh: {mouth_debug_values['threshold']:.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 0, 255), 1)
                y_offset += 25
        
        # Instructions
        if self.config.use_relative_movement:
            instructions = [
                "ESC: Exit | D: Debug | M: Mesh | N: Circle | R: Toggle Mode",
                "OPEN MOUTH: Move nose to control cursor (velocity)",
                "CLOSE MOUTH: Lock cursor & recalibrate center",
                "Blink both: Toggle scroll | Wink: Click (when mouth closed)"
            ]
        else:
            instructions = [
                "ESC: Exit | D: Debug | M: Mesh | R: Toggle Mode",
                "OPEN MOUTH: Move cursor (blocks clicks)",
                "CLOSE MOUTH: Lock cursor at position",
                "Blink both: Toggle scroll | Wink: Click (when mouth closed)"
            ]
        
        y_pos = h - 80
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("\n" + "="*70)
        print("MOUTH-CONTROLLED VIRTUAL MOUSE")
        print("="*70)
        if self.config.use_relative_movement:
            print("🎯 MODE: RELATIVE (Nose velocity-based)")
            print("👄 OPEN MOUTH:          Move nose relative to center")
            print("   - Small movements:   Slow cursor movement")
            print("   - Large movements:   Fast cursor movement")
            print("👄 CLOSE MOUTH:         Lock cursor & recalibrate center")
        else:
            print("🎯 MODE: ABSOLUTE (Iris position-based)")
            print("👄 OPEN MOUTH:          Move cursor (clicks blocked)")
            print("👄 CLOSE MOUTH:         Lock cursor at current position")
        print("👁️  Blink both eyes:     Toggle scroll mode (when mouth closed)")
        print("😉 Wink left/right:     Click (when mouth closed)")
        print()
        print("⌨️  Keyboard shortcuts:")
        print("   ESC - Exit program")
        print("   R   - Toggle between Relative/Absolute mode")
        print("   D   - Toggle debug values")
        print("   M   - Toggle face mesh")
        print("   N   - Toggle nose circle (relative mode)")
        print("   S   - Toggle statistics")
        print("   +/- - Adjust mouth sensitivity")
        print("   [/] - Adjust nose speed (relative mode)")
        print("   C   - Save configuration")
        print("="*70 + "\n")
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            try:
                while True:
                    success, frame = cap.read()
                    
                    if not success:
                        print("⚠️  Warning: Empty camera frame")
                        continue
                    
                    # Flip for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    nose_debug = None
                    
                    # Process face landmarks
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        landmarks = face_landmarks.landmark
                        
                        # Get debug values
                        eye_debug_values = self.gesture_detector.get_debug_values(landmarks)
                        mouth_debug_values = self.mouth_detector.get_debug_values(landmarks)
                        
                        # Detect mouth open/closed
                        mouth_is_open = self.mouth_detector.detect_mouth_open(landmarks)
                        
                        # Update movement state
                        was_moving = self.is_moving
                        self.is_moving = mouth_is_open
                        
                        # CURSOR MOVEMENT
                        if self.config.use_relative_movement:
                            # === RELATIVE MODE (Nose-based velocity) ===
                            if self.is_moving:
                                # Move cursor based on nose displacement
                                moved = self.nose_controller.move_cursor(landmarks)
                                
                                if not was_moving:
                                    print("🖱️  Relative movement: ACTIVE")
                            else:
                                # Mouth closed - update reference position
                                self.nose_controller.update_reference_position(landmarks)
                                
                                if was_moving:
                                    print("🔒 Cursor locked, center recalibrated")
                            
                            # Get debug info for visualization
                            nose_debug = self.nose_controller.get_debug_info()
                        
                        else:
                            # === ABSOLUTE MODE (Iris-based position) ===
                            if self.is_moving:
                                # Move cursor to iris position
                                cursor_x, cursor_y = self.calculate_cursor_position_absolute(
                                    landmarks, frame.shape)
                                pag.moveTo(cursor_x, cursor_y, duration=0, _pause=False)
                                self.last_cursor_pos = (cursor_x, cursor_y)
                                
                                if not was_moving:
                                    self.cursor_filter.reset()
                                    print("🖱️  Cursor movement: ACTIVE")
                            else:
                                # Mouth closed - cursor stays at last position
                                if was_moving:
                                    print("🔒 Cursor locked at position")
                        
                        # GESTURE DETECTION - Only when NOT moving (mouth closed)
                        if not self.is_moving:
                            # 1. Blink - toggle scroll mode
                            if self.gesture_detector.detect_blink(landmarks):
                                self.scroll_mode = not self.scroll_mode
                                print(f"{'✓' if self.scroll_mode else '✗'} Scroll mode: "
                                      f"{'ON' if self.scroll_mode else 'OFF'}")
                            
                            # 2. Wink - click
                            wink = self.gesture_detector.detect_wink(landmarks)
                            if wink:
                                button = 'left' if wink == 'left' else 'right'
                                pag.click(button=button)
                                print(f"🖱️  {button.capitalize()} click")
                        
                        # Draw visualizations
                        self.draw_visualizations(frame, face_landmarks, 
                                               eye_debug_values, mouth_debug_values, nose_debug)
                    else:
                        # No face detected
                        cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.is_moving = False
                    
                    # Calculate FPS
                    self.frame_count += 1
                    if self.frame_count % 10 == 0:
                        elapsed = time.time() - self.fps_start_time
                        self.fps = self.frame_count / elapsed
                        if elapsed > 2.0:
                            self.frame_count = 0
                            self.fps_start_time = time.time()
                    
                    # Display frame
                    cv2.imshow('Mouth-Controlled Virtual Mouse', frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 27:  # ESC
                        print("\n👋 Exiting...")
                        break
                    elif key == ord('r') or key == ord('R'):
                        # Toggle between relative and absolute mode
                        self.config.use_relative_movement = not self.config.use_relative_movement
                        
                        if self.config.use_relative_movement:
                            # Switch to relative mode
                            if not hasattr(self, 'nose_controller'):
                                self.nose_controller = NoseRelativeController(
                                    self.config, (self.screen_w, self.screen_h))
                            else:
                                self.nose_controller.reset()
                            print("✓ Switched to RELATIVE mode (nose velocity)")
                        else:
                            # Switch to absolute mode
                            if not hasattr(self, 'cursor_filter'):
                                self.cursor_filter = SmoothingFilter(self.config.cursor_smoothing)
                            else:
                                self.cursor_filter.reset()
                            print("✓ Switched to ABSOLUTE mode (iris position)")
                    
                    elif key == ord('d') or key == ord('D'):
                        self.config.show_debug_values = not self.config.show_debug_values
                        print(f"Debug values: {'ON' if self.config.show_debug_values else 'OFF'}")
                    elif key == ord('m') or key == ord('M'):
                        self.config.show_face_mesh = not self.config.show_face_mesh
                        print(f"Face mesh: {'ON' if self.config.show_face_mesh else 'OFF'}")
                    elif key == ord('n') or key == ord('N'):
                        self.config.show_nose_circle = not self.config.show_nose_circle
                        print(f"Nose circle: {'ON' if self.config.show_nose_circle else 'OFF'}")
                    elif key == ord('s') or key == ord('S'):
                        self.config.show_stats = not self.config.show_stats
                        print(f"Statistics: {'ON' if self.config.show_stats else 'OFF'}")
                    elif key == ord('=') or key == ord('+'):
                        self.config.mouth_open_thresh -= 0.005
                        self.config.mouth_open_thresh = max(0.01, self.config.mouth_open_thresh)
                        print(f"Mouth sensitivity increased: {self.config.mouth_open_thresh:.3f}")
                    elif key == ord('-') or key == ord('_'):
                        self.config.mouth_open_thresh += 0.005
                        self.config.mouth_open_thresh = min(0.10, self.config.mouth_open_thresh)
                        print(f"Mouth sensitivity decreased: {self.config.mouth_open_thresh:.3f}")
                    elif key == ord('['):
                        if self.config.use_relative_movement:
                            self.config.nose_speed_multiplier -= 2.0
                            self.config.nose_speed_multiplier = max(5.0, self.config.nose_speed_multiplier)
                            print(f"Nose speed decreased: {self.config.nose_speed_multiplier:.1f}")
                    elif key == ord(']'):
                        if self.config.use_relative_movement:
                            self.config.nose_speed_multiplier += 2.0
                            self.config.nose_speed_multiplier = min(50.0, self.config.nose_speed_multiplier)
                            print(f"Nose speed increased: {self.config.nose_speed_multiplier:.1f}")
                    elif key == ord('c') or key == ord('C'):
                        self.config.save()
                        print("✓ Configuration saved to mouse_config.json")
            
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
            except pag.FailSafeException:
                print("\n🛑 PyAutoGUI failsafe triggered (mouse in corner)")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                cap.release()
                cv2.destroyAllWindows()
                print("✓ Cleanup complete")

def headmouse():
    try:
        mouse = VirtualMouse()
        frame = mouse.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Entry point"""
    print("\n" + "="*70)
    print("MOUTH-CONTROLLED VIRTUAL MOUSE")
    print("Enhanced with Relative Nose Movement")
    print("="*70)
    print("Initializing detection systems...")
    
    try:
        mouse = VirtualMouse()
        mouse.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()