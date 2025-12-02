"""
Enhanced Virtual Mouse Control System - Mouth-Controlled Movement
Uses MediaPipe Face Mesh for robust face tracking with mouth-open detection
Cursor only moves when mouth is open, other actions blocked during movement
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
    
    # ============================================================================
    # CALIBRATED THRESHOLDS - Based on your facial measurements
    # ============================================================================
    
    # Eye detection thresholds
    eye_closure_thresh: float = 0.265  # ← UPDATED from 0.18
    blink_consecutive_frames: int = 2
    wink_ar_diff_thresh: float = 0.045  # ← UPDATED from 0.012
    wink_consecutive_frames: int = 6
    
    # Mouth detection thresholds
    mouth_open_thresh: float = 0.500  # ← UPDATED from 0.035
    mouth_consecutive_frames: int = 5
    
    # ============================================================================
    # Mouse control (unchanged)
    # ============================================================================
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
            
            # Filter out fields that aren't in the Config dataclass
            # This allows the JSON to have extra metadata fields
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
        # print(f"MAR: {mar:.4f}")
        # print(f"Threshold: {self.config.mouth_open_thresh:.4f}")
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
        
        # Both eyes should be closed for a blink
        avg_closure = (left_closure + right_closure) / 2.0
        
        if avg_closure > self.config.eye_closure_thresh:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.config.blink_consecutive_frames:
                # Debounce - prevent multiple detections
                current_time = time.time()
                if current_time - self.last_blink_time > 0.4:
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
    """Main virtual mouse controller - Mouth-controlled cursor movement"""
    
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
        self.cursor_filter = SmoothingFilter(self.config.cursor_smoothing)
        
        # State variables
        self.scroll_mode = False
        self.is_moving = False  # Track if cursor is currently moving
        self.last_cursor_pos = None  # Store last cursor position when mouth closes
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        print("✓ Mouth-Controlled Virtual Mouse initialized")
        print(f"✓ Screen resolution: {self.screen_w}x{self.screen_h}")
        print(f"✓ Mouth-open detection: ENABLED")
        print(f"✓ Multi-landmark tracking: ACTIVE")
    
    def calculate_cursor_position(self, landmarks, frame_shape) -> Tuple[float, float]:
        """Calculate cursor position from iris landmarks"""
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
    
    def draw_visualizations(self, frame, face_landmarks, eye_debug_values: dict, 
                           mouth_debug_values: dict):
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
        
        if self.config.show_stats:
            y_offset = 30
            
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
            
            # Debug values for calibration
            if self.config.show_debug_values:
                # Eye debug
                cv2.putText(frame, 
                           f"Eye L: {eye_debug_values['left_closure']:.3f} | "
                           f"R: {eye_debug_values['right_closure']:.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (255, 255, 0), 1)
                y_offset += 25
                
                # Mouth debug
                cv2.putText(frame, 
                           f"Mouth: {mouth_debug_values['mouth_ratio']:.3f} | "
                           f"Thresh: {mouth_debug_values['threshold']:.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 255), 1)
                y_offset += 25
        
        # Instructions
        instructions = [
            "ESC: Exit | D: Debug | M: Mesh | S: Stats",
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
        print("👄 OPEN MOUTH:          Move cursor (clicks blocked)")
        print("👄 CLOSE MOUTH:         Lock cursor at current position")
        print("👁️  Blink both eyes:     Toggle scroll mode (when mouth closed)")
        print("😉 Wink left/right:     Click (when mouth closed)")
        print()
        print("⌨️  Keyboard shortcuts:")
        print("   ESC - Exit program")
        print("   D   - Toggle debug values")
        print("   M   - Toggle face mesh")
        print("   S   - Toggle statistics")
        print("   +/- - Adjust mouth sensitivity")
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
                        
                        # CURSOR MOVEMENT - Only when mouth is open
                        if self.is_moving:
                            # Move cursor
                            cursor_x, cursor_y = self.calculate_cursor_position(
                                landmarks, frame.shape)
                            pag.moveTo(cursor_x, cursor_y, duration=0, _pause=False)
                            self.last_cursor_pos = (cursor_x, cursor_y)
                            
                            # Clear smoothing filter when starting movement
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
                                               eye_debug_values, mouth_debug_values)
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
                    elif key == ord('d') or key == ord('D'):
                        self.config.show_debug_values = not self.config.show_debug_values
                        print(f"Debug values: {'ON' if self.config.show_debug_values else 'OFF'}")
                    elif key == ord('m') or key == ord('M'):
                        self.config.show_face_mesh = not self.config.show_face_mesh
                        print(f"Face mesh: {'ON' if self.config.show_face_mesh else 'OFF'}")
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


def main():
    """Entry point"""
    print("\n" + "="*70)
    print("MOUTH-CONTROLLED VIRTUAL MOUSE")
    print("="*70)
    print("Initializing mouth detection system...")
    
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