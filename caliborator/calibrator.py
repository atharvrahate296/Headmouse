"""
Face Gesture Calibration Tool
Records facial gesture data to determine optimal detection thresholds
"""

import sys
import time
import json
from datetime import datetime
from collections import deque
from typing import List, Dict
import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError as e:
    print(f"❌ FATAL ERROR: Missing Required Library")
    print(f"Details: {e}")
    print("\nPlease install required libraries:")
    print("pip install opencv-python mediapipe")
    sys.exit(1)


class CalibrationRecorder:
    """Records facial gesture measurements for calibration"""
    
    # Eye landmarks
    LEFT_EYE_CORE = [362, 385, 387, 263, 373, 380]
    LEFT_EYELID_UPPER = [384, 398, 362]
    LEFT_EYELID_LOWER = [381, 380, 374]
    LEFT_EYEBROW = [276, 283, 282, 295, 285]
    
    RIGHT_EYE_CORE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYELID_UPPER = [160, 159, 158]
    RIGHT_EYELID_LOWER = [144, 145, 153]
    RIGHT_EYEBROW = [46, 53, 52, 65, 55]
    
    # Mouth landmarks
    MOUTH_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    MOUTH_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    MOUTH_LEFT = [61, 146]
    MOUTH_RIGHT = [291, 375]
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'gestures': []
        }
        
    def calculate_enhanced_ear(self, eye_core_landmarks, landmarks) -> float:
        """Calculate Eye Aspect Ratio"""
        points = np.array([(landmarks[i].x, landmarks[i].y, landmarks[i].z) 
                          for i in eye_core_landmarks])
        
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        h = np.linalg.norm(points[0] - points[3])
        
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def calculate_eyelid_distance(self, upper_landmarks, lower_landmarks, landmarks) -> float:
        """Calculate vertical distance between eyelids"""
        upper_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in upper_landmarks])
        lower_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in lower_landmarks])
        
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
        
        ratio = abs(eyebrow_y - eye_y)
        return ratio
    
    def calculate_comprehensive_eye_closure(self, eye_core, eyelid_upper, 
                                           eyelid_lower, eyebrow, landmarks) -> Dict:
        """Calculate comprehensive eye closure metrics"""
        ear = self.calculate_enhanced_ear(eye_core, landmarks)
        ear_score = 1.0 - (ear * 15)
        ear_score = np.clip(ear_score, 0, 1)
        
        eyelid_dist = self.calculate_eyelid_distance(eyelid_upper, eyelid_lower, landmarks)
        eyelid_score = 1.0 - (eyelid_dist * 30)
        eyelid_score = np.clip(eyelid_score, 0, 1)
        
        eyebrow_ratio = self.calculate_eyebrow_eye_ratio(eyebrow, eye_core, landmarks)
        eyebrow_score = 1.0 - (eyebrow_ratio * 10)
        eyebrow_score = np.clip(eyebrow_score, 0, 1)
        
        combined = (ear_score * 0.5 + eyelid_score * 0.35 + eyebrow_score * 0.15)
        
        return {
            'ear': ear,
            'ear_score': ear_score,
            'eyelid_distance': eyelid_dist,
            'eyelid_score': eyelid_score,
            'eyebrow_ratio': eyebrow_ratio,
            'eyebrow_score': eyebrow_score,
            'combined_closure': combined
        }
    
    def calculate_mouth_aspect_ratio(self, landmarks) -> Dict:
        """Calculate Mouth Aspect Ratio and related metrics"""
        upper_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in self.MOUTH_UPPER])
        lower_points = np.array([(landmarks[i].x, landmarks[i].y) 
                                for i in self.MOUTH_LOWER])
        
        # Calculate vertical distances
        vertical_dists = []
        for i in range(min(len(upper_points), len(lower_points))):
            dist = np.linalg.norm(upper_points[i] - lower_points[i])
            vertical_dists.append(dist)
        
        avg_vertical = np.mean(vertical_dists)
        max_vertical = np.max(vertical_dists)
        
        # Calculate horizontal distance
        left_point = np.array([landmarks[self.MOUTH_LEFT[0]].x, 
                              landmarks[self.MOUTH_LEFT[0]].y])
        right_point = np.array([landmarks[self.MOUTH_RIGHT[0]].x, 
                               landmarks[self.MOUTH_RIGHT[0]].y])
        horizontal = np.linalg.norm(right_point - left_point)
        
        # MAR = vertical / horizontal
        mar = avg_vertical / (horizontal + 1e-6)
        mar_max = max_vertical / (horizontal + 1e-6)
        
        return {
            'mouth_aspect_ratio': mar,
            'mar_max': mar_max,
            'avg_vertical': avg_vertical,
            'max_vertical': max_vertical,
            'horizontal': horizontal
        }
    
    def get_all_measurements(self, landmarks) -> Dict:
        """Get all facial measurements"""
        left_eye = self.calculate_comprehensive_eye_closure(
            self.LEFT_EYE_CORE, self.LEFT_EYELID_UPPER,
            self.LEFT_EYELID_LOWER, self.LEFT_EYEBROW, landmarks
        )
        
        right_eye = self.calculate_comprehensive_eye_closure(
            self.RIGHT_EYE_CORE, self.RIGHT_EYELID_UPPER,
            self.RIGHT_EYELID_LOWER, self.RIGHT_EYEBROW, landmarks
        )
        
        mouth = self.calculate_mouth_aspect_ratio(landmarks)
        
        return {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'mouth': mouth,
            'avg_eye_closure': (left_eye['combined_closure'] + right_eye['combined_closure']) / 2.0,
            'eye_closure_diff': abs(left_eye['combined_closure'] - right_eye['combined_closure'])
        }
    
    def record_gesture(self, gesture_name: str, duration: float = 2.0):
        """Record a specific gesture for a duration"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return None
        
        print(f"\n{'='*70}")
        print(f"RECORDING: {gesture_name}")
        print(f"{'='*70}")
        print(f"Duration: {duration} seconds")
        print("Press SPACE when ready to start recording...")
        print("Press ESC to skip this gesture")
        print(f"{'='*70}\n")
        
        measurements = []
        recording = False
        start_time = None
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            while True:
                success, frame = cap.read()
                
                if not success:
                    continue
                
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                # Draw status
                if recording:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    
                    if remaining <= 0:
                        print(f"✓ Recording complete for {gesture_name}")
                        break
                    
                    # Recording indicator
                    cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(frame, f"RECORDING: {remaining:.1f}s", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Collect measurements
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        landmarks = face_landmarks.landmark
                        
                        measurement = self.get_all_measurements(landmarks)
                        measurements.append(measurement)
                        
                        # Display current values
                        y_pos = 80
                        cv2.putText(frame, f"Left Eye: {measurement['left_eye']['combined_closure']:.3f}",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y_pos += 30
                        cv2.putText(frame, f"Right Eye: {measurement['right_eye']['combined_closure']:.3f}",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y_pos += 30
                        cv2.putText(frame, f"Mouth: {measurement['mouth']['mouth_aspect_ratio']:.3f}",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    # Waiting to start
                    cv2.putText(frame, f"GET READY: {gesture_name}", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to start recording", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Press ESC to skip", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    
                    # Show live preview
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        landmarks = face_landmarks.landmark
                        
                        measurement = self.get_all_measurements(landmarks)
                        
                        y_pos = 160
                        cv2.putText(frame, "LIVE PREVIEW:", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_pos += 30
                        cv2.putText(frame, f"Left Eye: {measurement['left_eye']['combined_closure']:.3f}",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_pos += 25
                        cv2.putText(frame, f"Right Eye: {measurement['right_eye']['combined_closure']:.3f}",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_pos += 25
                        cv2.putText(frame, f"Mouth: {measurement['mouth']['mouth_aspect_ratio']:.3f}",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Calibration Recorder', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print(f"⚠️  Skipped: {gesture_name}")
                    cap.release()
                    return None
                elif key == 32 and not recording:  # SPACE
                    recording = True
                    start_time = time.time()
                    print(f"🔴 Recording started for {gesture_name}...")
        
        cap.release()
        
        # Calculate statistics
        if measurements:
            return self.calculate_statistics(gesture_name, measurements)
        return None
    
    def calculate_statistics(self, gesture_name: str, measurements: List[Dict]) -> Dict:
        """Calculate statistics from recorded measurements"""
        stats = {
            'gesture_name': gesture_name,
            'sample_count': len(measurements),
            'left_eye': {},
            'right_eye': {},
            'mouth': {},
            'avg_eye_closure': {},
            'eye_closure_diff': {}
        }
        
        # Extract arrays for each metric
        for key in ['left_eye', 'right_eye', 'mouth', 'avg_eye_closure', 'eye_closure_diff']:
            if key in ['left_eye', 'right_eye']:
                for metric in ['combined_closure', 'ear', 'eyelid_distance']:
                    values = [m[key][metric] for m in measurements]
                    stats[key][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values))
                    }
            elif key == 'mouth':
                for metric in ['mouth_aspect_ratio', 'mar_max', 'avg_vertical']:
                    values = [m[key][metric] for m in measurements]
                    stats[key][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values))
                    }
            else:
                values = [m[key] for m in measurements]
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return stats
    
    def run_calibration(self):
        """Run full calibration sequence"""
        gestures = [
            ("neutral", "Keep your face neutral/relaxed", 3.0),
            ("mouth_open", "Open your mouth (like saying 'Ahhh')", 3.0),
            ("mouth_closed", "Keep your mouth firmly closed", 3.0),
            ("eyes_open", "Keep both eyes wide open", 3.0),
            ("both_eyes_closed", "Close both eyes (blink and hold)", 3.0),
            ("left_eye_closed", "Close/wink your LEFT eye only", 3.0),
            ("right_eye_closed", "Close/wink your RIGHT eye only", 3.0),
            ("slight_mouth_open", "Open mouth slightly (small gap)", 2.0),
        ]
        
        print("\n" + "="*70)
        print("FACIAL GESTURE CALIBRATION TOOL")
        print("="*70)
        print("This tool will record your facial gestures to find optimal thresholds.")
        print("You will be prompted to perform different gestures.")
        print("Try to hold each gesture steady during the recording.")
        print("="*70)
        
        input("\nPress ENTER to begin calibration...")
        
        for gesture_name, description, duration in gestures:
            print(f"\n📋 Next gesture: {description}")
            input("Press ENTER when ready...")
            
            stats = self.record_gesture(gesture_name, duration)
            
            if stats:
                self.calibration_data['gestures'].append(stats)
                print(f"✓ {gesture_name} recorded successfully")
            else:
                print(f"⚠️  {gesture_name} was skipped")
            
            time.sleep(0.5)
        
        # Save results
        cv2.destroyAllWindows()
        self.save_calibration_data()
        self.print_summary()
    
    def save_calibration_data(self):
        """Save calibration data to JSON file"""
        filename = f"calibration_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        
        print(f"\n✓ Calibration data saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print summary of calibration results"""
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY")
        print("="*70)
        
        for gesture in self.calibration_data['gestures']:
            name = gesture['gesture_name']
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  Samples: {gesture['sample_count']}")
            
            if 'mouth' in gesture:
                mar = gesture['mouth']['mouth_aspect_ratio']
                print(f"  Mouth Ratio: {mar['mean']:.4f} (±{mar['std']:.4f}) "
                      f"[{mar['min']:.4f} - {mar['max']:.4f}]")
            
            if 'avg_eye_closure' in gesture:
                eye = gesture['avg_eye_closure']
                print(f"  Eye Closure: {eye['mean']:.4f} (±{eye['std']:.4f}) "
                      f"[{eye['min']:.4f} - {eye['max']:.4f}]")
            
            if 'eye_closure_diff' in gesture:
                diff = gesture['eye_closure_diff']
                print(f"  Eye Diff:    {diff['mean']:.4f} (±{diff['std']:.4f}) "
                      f"[{diff['min']:.4f} - {diff['max']:.4f}]")
        
        print("\n" + "="*70)
        print("RECOMMENDED THRESHOLDS:")
        print("="*70)
        
        # Calculate recommended thresholds
        try:
            # Find gestures
            mouth_open = next((g for g in self.calibration_data['gestures'] 
                             if g['gesture_name'] == 'mouth_open'), None)
            mouth_closed = next((g for g in self.calibration_data['gestures'] 
                               if g['gesture_name'] == 'mouth_closed'), None)
            eyes_open = next((g for g in self.calibration_data['gestures'] 
                            if g['gesture_name'] == 'eyes_open'), None)
            eyes_closed = next((g for g in self.calibration_data['gestures'] 
                              if g['gesture_name'] == 'both_eyes_closed'), None)
            
            if mouth_open and mouth_closed:
                mouth_open_val = mouth_open['mouth']['mouth_aspect_ratio']['mean']
                mouth_closed_val = mouth_closed['mouth']['mouth_aspect_ratio']['mean']
                mouth_threshold = (mouth_open_val + mouth_closed_val) / 2
                print(f"\nMouth Open Threshold: {mouth_threshold:.4f}")
                print(f"  (between {mouth_closed_val:.4f} closed and {mouth_open_val:.4f} open)")
            
            if eyes_open and eyes_closed:
                eyes_open_val = eyes_open['avg_eye_closure']['mean']
                eyes_closed_val = eyes_closed['avg_eye_closure']['mean']
                eye_threshold = (eyes_open_val + eyes_closed_val) / 2
                print(f"\nEye Closure Threshold: {eye_threshold:.4f}")
                print(f"  (between {eyes_open_val:.4f} open and {eyes_closed_val:.4f} closed)")
            
        except Exception as e:
            print(f"\n⚠️  Could not calculate recommended thresholds: {e}")
        
        print("\n" + "="*70)


def main():
    """Entry point"""
    try:
        recorder = CalibrationRecorder()
        recorder.run_calibration()
        
        print("\n✓ Calibration complete!")
        print("Share the generated JSON file to get optimized detection thresholds.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Calibration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()