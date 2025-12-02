import sys
import time
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Deque
import numpy as np
import math

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
    eye_closure_thresh: float = 0.265
    blink_consecutive_frames: int = 2
    wink_ar_diff_thresh: float = 0.045
    wink_consecutive_frames: int = 6
    
    # Mouth detection thresholds
    mouth_open_thresh: float = 0.500
    mouth_consecutive_frames: int = 5
    
    # ============================================================================
    # JOYSTICK / BOX CONTROL SETTINGS (NEW)
    # ============================================================================
    # The size of the "Dead Zone" box. 
    # Cursor won't move if nose is within this distance from anchor.
    box_width: int = 35   # Horizontal deadzone radius
    box_height: int = 25  # Vertical deadzone radius
    cursor_speed: int = 15 # Pixels per frame to move when active
    
    # Camera settings
    cam_width: int = 640
    cam_height: int = 480
    
    # Visual feedback
    show_face_mesh: bool = True
    show_eye_markers: bool = True
    show_mouth_markers: bool = True
    show_box_overlay: bool = True  # NEW: Show the joystick box
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
            return cls()


class SmoothingFilter:
    """Exponential moving average filter for smooth coordinates"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.alpha = 2 / (window_size + 1)
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        self.history.append((x, y))
        if len(self.history) == 1:
            return x, y
        
        # EMA
        prev_x, prev_y = self.history[-2] if len(self.history) > 1 else (x, y)
        smooth_x = self.alpha * x + (1 - self.alpha) * prev_x
        smooth_y = self.alpha * y + (1 - self.alpha) * prev_y
        
        return smooth_x, smooth_y
    
    def reset(self):
        self.history.clear()


class MouthDetector:
    """Detects mouth open/closed state for cursor movement control"""
    
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
        upper_points = np.array([(landmarks[i].x, landmarks[i].y) for i in self.MOUTH_UPPER])
        lower_points = np.array([(landmarks[i].x, landmarks[i].y) for i in self.MOUTH_LOWER])
        
        vertical_dists = []
        for i in range(min(len(upper_points), len(lower_points))):
            dist = np.linalg.norm(upper_points[i] - lower_points[i])
            vertical_dists.append(dist)
        
        avg_vertical = np.mean(vertical_dists)
        
        left_point = np.array([landmarks[self.MOUTH_LEFT[0]].x, landmarks[self.MOUTH_LEFT[0]].y])
        right_point = np.array([landmarks[self.MOUTH_RIGHT[0]].x, landmarks[self.MOUTH_RIGHT[0]].y])
        horizontal = np.linalg.norm(right_point - left_point)
        
        return avg_vertical / (horizontal + 1e-6)
    
    def detect_mouth_open(self, landmarks) -> bool:
        mar = self.calculate_mouth_aspect_ratio(landmarks)
        self.mouth_ratio_history.append(mar)
        
        if len(self.mouth_ratio_history) > 1:
            mar = np.mean(self.mouth_ratio_history)
            
        if mar/10 > self.config.mouth_open_thresh: 
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
        mar = self.calculate_mouth_aspect_ratio(landmarks)
        return {
            'mouth_ratio': mar,
            'is_open': self.is_mouth_open,
            'threshold': self.config.mouth_open_thresh
        }


class EnhancedGestureDetector:
    """Advanced gesture detector using multiple landmarks around eyes"""
    
    LEFT_EYE_CORE = [362, 385, 387, 263, 373, 380]
    LEFT_EYE_OUTER = [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    LEFT_EYEBROW = [276, 283, 282, 295, 285]
    LEFT_EYELID_UPPER = [384, 398, 362]
    LEFT_EYELID_LOWER = [381, 380, 374]
    
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
        self.left_closure_history = deque(maxlen=3)
        self.right_closure_history = deque(maxlen=3)
    
    def calculate_enhanced_ear(self, eye_core_landmarks, landmarks) -> float:
        points = np.array([(landmarks[i].x, landmarks[i].y, landmarks[i].z) 
                          for i in eye_core_landmarks])
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        h = np.linalg.norm(points[0] - points[3])
        return (v1 + v2) / (2.0 * h + 1e-6)
    
    def calculate_eyelid_distance(self, upper_landmarks, lower_landmarks, landmarks) -> float:
        upper_points = np.array([(landmarks[i].x, landmarks[i].y) for i in upper_landmarks])
        lower_points = np.array([(landmarks[i].x, landmarks[i].y) for i in lower_landmarks])
        return np.linalg.norm(np.mean(upper_points, axis=0) - np.mean(lower_points, axis=0))
    
    def calculate_eyebrow_eye_ratio(self, eyebrow_landmarks, eye_landmarks, landmarks) -> float:
        eyebrow_points = np.array([(landmarks[i].x, landmarks[i].y) for i in eyebrow_landmarks])
        eye_points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_landmarks])
        return abs(np.mean(eyebrow_points[:, 1]) - np.mean(eye_points[:, 1]))
    
    def calculate_comprehensive_eye_closure(self, eye_core, eyelid_upper, 
                                           eyelid_lower, eyebrow, landmarks) -> float:
        ear = self.calculate_enhanced_ear(eye_core, landmarks)
        ear_score = np.clip(1.0 - (ear * 15), 0, 1)
        eyelid_dist = self.calculate_eyelid_distance(eyelid_upper, eyelid_lower, landmarks)
        eyelid_score = np.clip(1.0 - (eyelid_dist * 30), 0, 1)
        eyebrow_ratio = self.calculate_eyebrow_eye_ratio(eyebrow, eye_core, landmarks)
        eyebrow_score = np.clip(1.0 - (eyebrow_ratio * 10), 0, 1)
        return (ear_score * 0.5 + eyelid_score * 0.35 + eyebrow_score * 0.15)
    
    def detect_blink(self, landmarks) -> bool:
        left_closure = self.calculate_comprehensive_eye_closure(
            self.LEFT_EYE_CORE, self.LEFT_EYELID_UPPER, 
            self.LEFT_EYELID_LOWER, self.LEFT_EYEBROW, landmarks)
        right_closure = self.calculate_comprehensive_eye_closure(
            self.RIGHT_EYE_CORE, self.RIGHT_EYELID_UPPER,
            self.RIGHT_EYELID_LOWER, self.RIGHT_EYEBROW, landmarks)
        
        self.left_closure_history.append(left_closure)
        self.right_closure_history.append(right_closure)
        
        if len(self.left_closure_history) > 1:
            left_closure = np.mean(self.left_closure_history)
            right_closure = np.mean(self.right_closure_history)
        
        avg_closure = (left_closure + right_closure) / 2.0
        
        if avg_closure > self.config.eye_closure_thresh:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.config.blink_consecutive_frames:
                current_time = time.time()
                if current_time - self.last_blink_time > 0.4:
                    self.blink_counter = 0
                    self.last_blink_time = current_time
                    return True
            self.blink_counter = 0
        return False
    
    def detect_wink(self, landmarks) -> Optional[str]:
        left_closure = self.calculate_comprehensive_eye_closure(
            self.LEFT_EYE_CORE, self.LEFT_EYELID_UPPER,
            self.LEFT_EYELID_LOWER, self.LEFT_EYEBROW, landmarks)
        right_closure = self.calculate_comprehensive_eye_closure(
            self.RIGHT_EYE_CORE, self.RIGHT_EYELID_UPPER,
            self.RIGHT_EYELID_LOWER, self.RIGHT_EYEBROW, landmarks)
            
        self.left_closure_history.append(left_closure)
        self.right_closure_history.append(right_closure)
        
        if len(self.left_closure_history) > 1:
            left_closure = np.mean(self.left_closure_history)
            right_closure = np.mean(self.right_closure_history)
            
        closure_diff = abs(left_closure - right_closure)
        
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
        left_closure = self.calculate_comprehensive_eye_closure(
            self.LEFT_EYE_CORE, self.LEFT_EYELID_UPPER,
            self.LEFT_EYELID_LOWER, self.LEFT_EYEBROW, landmarks)
        right_closure = self.calculate_comprehensive_eye_closure(
            self.RIGHT_EYE_CORE, self.RIGHT_EYELID_UPPER,
            self.RIGHT_EYELID_LOWER, self.RIGHT_EYEBROW, landmarks)
        return {
            'left_closure': left_closure,
            'right_closure': right_closure,
            'avg_closure': (left_closure + right_closure) / 2.0,
            'closure_diff': abs(left_closure - right_closure)
        }


class VirtualMouse:
    """Main virtual mouse controller - Box/Joystick Movement Logic"""
    
    # Landmark 4 is the tip of the nose
    NOSE_TIP = 4
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        
        # PyAutoGUI safety
        pag.FAILSAFE = True
        pag.PAUSE = 0.001
        self.screen_w, self.screen_h = pag.size()
        
        # Components
        self.gesture_detector = EnhancedGestureDetector(self.config)
        self.mouth_detector = MouthDetector(self.config)
        self.nose_filter = SmoothingFilter(3) # Low smoothing for nose to keep response snappy
        
        # State variables
        self.scroll_mode = False
        self.is_moving = False 
        
        # JOYSTICK STATE
        self.anchor_point = None  # (x, y) coordinates when mouth opens
        self.current_nose_pos = None # (x, y) current nose coordinates
        
        # FPS and timing
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        print("✓ Box-Control Virtual Mouse initialized")
        print(f"✓ Control Method: Joystick/Box (Relative)")
        print(f"✓ Mouth-open detection: ENABLED (Hold to Move)")
    
    def process_movement_logic(self, frame_shape):
        """
        Implements the Box/Joystick logic:
        1. If mouth just opened, set Anchor.
        2. If nose outside box, move cursor relative to direction.
        """
        if self.current_nose_pos is None:
            return

        nx, ny = self.current_nose_pos

        if self.is_moving:
            # RISING EDGE: Mouth just opened, set ANCHOR
            if self.anchor_point is None:
                self.anchor_point = (nx, ny)
                self.nose_filter.reset()
                print("⚓ Anchor set!")
            
            # JOYSTICK LOGIC
            ax, ay = self.anchor_point
            
            # Calculate distance from anchor
            diff_x = nx - ax
            diff_y = ny - ay
            
            move_x = 0
            move_y = 0
            
            # Horizontal movement
            if abs(diff_x) > self.config.box_width:
                if diff_x > 0: # Looking Right
                    move_x = self.config.cursor_speed
                else: # Looking Left
                    move_x = -self.config.cursor_speed
            
            # Vertical movement
            if abs(diff_y) > self.config.box_height:
                if diff_y > 0: # Looking Down
                    move_y = self.config.cursor_speed
                else: # Looking Up
                    move_y = -self.config.cursor_speed
            
            # Execute movement if outside deadzone
            if move_x != 0 or move_y != 0:
                if self.scroll_mode:
                    if move_y < 0: pag.scroll(40)
                    elif move_y > 0: pag.scroll(-40)
                else:
                    pag.moveRel(move_x, move_y)
                    
        else:
            # FALLING EDGE: Mouth closed, release anchor
            if self.anchor_point is not None:
                print("⚓ Anchor released")
                self.anchor_point = None

    def draw_visualizations(self, frame, face_landmarks, eye_debug, mouth_debug):
        """Draw visual feedback including the Joystick Box"""
        h, w = frame.shape[:2]
        
        if self.config.show_face_mesh:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        # DRAW JOYSTICK BOX
        if self.config.show_box_overlay and self.is_moving and self.anchor_point:
            ax, ay = self.anchor_point
            nx, ny = self.current_nose_pos
            
            bw = self.config.box_width
            bh = self.config.box_height
            
            # Draw Anchor Box (Deadzone) - GREEN
            cv2.rectangle(frame, (int(ax - bw), int(ay - bh)), 
                         (int(ax + bw), int(ay + bh)), (0, 255, 0), 2)
            
            # Draw Line from Anchor to Nose - BLUE
            cv2.line(frame, (int(ax), int(ay)), (int(nx), int(ny)), (255, 0, 0), 2)
            
            # Draw Current Nose Position - YELLOW
            cv2.circle(frame, (int(nx), int(ny)), 5, (0, 255, 255), -1)
            
            # Direction Text
            dir_text = ""
            if abs(nx - ax) > bw: dir_text += "RIGHT " if nx > ax else "LEFT "
            if abs(ny - ay) > bh: dir_text += "DOWN " if ny > ay else "UP "
            if dir_text:
                cv2.putText(frame, dir_text, (int(ax) - 20, int(ay) - bh - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Status Text (same as before)
        y_offset = 30
        status_color = (0, 255, 0) if not self.is_moving else (0, 255, 255)
        status_text = "CURSOR: LOCKED" if not self.is_moving else "JOYSTICK: ACTIVE"
        cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        y_offset += 30
        if self.scroll_mode:
            cv2.putText(frame, "SCROLL MODE", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)
        
        print("\n" + "="*70)
        print("MOUTH-CONTROLLED JOYSTICK MOUSE")
        print("="*70)
        print("👄 HOLD MOUTH OPEN:  Set Anchor & Activate Joystick")
        print("   -> Move Head Outside Box to move cursor")
        print("👄 CLOSE MOUTH:      Stop/Release Anchor")
        print("👁️  Blink/Wink:      Same as before (when mouth closed)")
        print("="*70 + "\n")
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            try:
                while True:
                    success, frame = cap.read()
                    if not success: continue
                    
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        landmarks = face_landmarks.landmark
                        
                        # 1. Update Nose Position (Landmark 4)
                        nose_lm = landmarks[self.NOSE_TIP]
                        # Apply slight smoothing to input coordinate to reduce jitter
                        nx, ny = self.nose_filter.update(nose_lm.x * w, nose_lm.y * h)
                        self.current_nose_pos = (nx, ny)
                        
                        # 2. Get Debug Data
                        eye_debug = self.gesture_detector.get_debug_values(landmarks)
                        mouth_debug = self.mouth_detector.get_debug_values(landmarks)
                        
                        # 3. Detect Mouth State
                        mouth_is_open = self.mouth_detector.detect_mouth_open(landmarks)
                        self.is_moving = mouth_is_open
                        
                        # 4. Process Box/Joystick Movement
                        self.process_movement_logic((h, w))
                        
                        # 5. Process Clicks/Scroll (Only when NOT moving/Mouth Closed)
                        if not self.is_moving:
                            if self.gesture_detector.detect_blink(landmarks):
                                self.scroll_mode = not self.scroll_mode
                            
                            wink = self.gesture_detector.detect_wink(landmarks)
                            if wink:
                                button = 'left' if wink == 'left' else 'right'
                                pag.click(button=button)
                                print(f"🖱️  {button.capitalize()} click")
                        
                        self.draw_visualizations(frame, face_landmarks, eye_debug, mouth_debug)
                        
                    cv2.imshow('Box-Control Mouse', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: break
                    elif key == ord('c'): self.config.save()
                    
            finally:
                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        VirtualMouse().run()
    except Exception as e:
        print(f"Error: {e}")