"""
Personalized Configuration for Virtual Mouse Control
Generated from calibration data: 2025-11-30T14:49:11
"""

from dataclasses import dataclass
import json

@dataclass
class CalibratedConfig:
    """
    Configuration optimized for your specific facial measurements
    
    Calibration Summary:
    - Eyes open avg: 0.230
    - Eyes closed avg: 0.331
    - Neutral mouth: 0.038
    - Mouth open avg: 0.076
    """
    
    # ============================================================================
    # EYE DETECTION THRESHOLDS (Calibrated)
    # ============================================================================
    
    # Combined eye closure threshold
    # Your range: open=0.230, closed=0.331
    # Setting: midpoint with safety margin
    eye_closure_thresh: float = 0.265  # ← CALIBRATED (was 0.18)
    
    # Frames to confirm blink
    blink_consecutive_frames: int = 2
    
    # Eye closure difference for wink detection
    # Your wink differences: left=0.161, right=0.087
    # Setting: well below smallest wink for reliability
    wink_ar_diff_thresh: float = 0.045  # ← CALIBRATED (was 0.012)
    
    # Frames to confirm wink
    wink_consecutive_frames: int = 6
    
    # ============================================================================
    # MOUTH DETECTION THRESHOLDS (Calibrated)
    # ============================================================================
    
    # Mouth opening threshold for cursor movement
    # Your range: closed=0.038, slight=0.057, open=0.076
    # Setting: between closed and slight open
    mouth_open_thresh: float = 0.050  # ← CALIBRATED (was 0.035)
    
    # Frames to confirm mouth state change
    mouth_consecutive_frames: int = 2
    
    # ============================================================================
    # MOUSE CONTROL SETTINGS (Keep existing)
    # ============================================================================
    
    # Smoothing filter window
    cursor_smoothing: int = 5
    
    # Movement sensitivity multiplier
    movement_multiplier: float = 1.8
    
    # Dead zone to prevent micro-movements
    dead_zone: float = 0.02
    
    # Edge margin (pixels from screen edge)
    edge_margin: int = 50
    
    # ============================================================================
    # CAMERA SETTINGS (Keep existing)
    # ============================================================================
    
    cam_width: int = 640
    cam_height: int = 480
    
    # ============================================================================
    # VISUAL FEEDBACK SETTINGS (Keep existing)
    # ============================================================================
    
    show_face_mesh: bool = True
    show_eye_markers: bool = True
    show_mouth_markers: bool = True
    show_stats: bool = True
    show_debug_values: bool = False
    
    # ============================================================================
    # CALIBRATION METADATA
    # ============================================================================
    
    calibration_date: str = "2025-11-30T14:49:11"
    calibration_version: str = "1.0"
    
    # Your personal ranges for reference
    eye_open_range: tuple = (0.220, 0.244)
    eye_closed_range: tuple = (0.300, 0.394)
    mouth_closed_range: tuple = (0.033, 0.044)
    mouth_open_range: tuple = (0.054, 0.092)
    
    def save(self, path: str = "mouse_config.json"):
        """Save configuration to file"""
        config_dict = {
            # Main thresholds
            "eye_closure_thresh": self.eye_closure_thresh,
            "blink_consecutive_frames": self.blink_consecutive_frames,
            "wink_ar_diff_thresh": self.wink_ar_diff_thresh,
            "wink_consecutive_frames": self.wink_consecutive_frames,
            "mouth_open_thresh": self.mouth_open_thresh,
            "mouth_consecutive_frames": self.mouth_consecutive_frames,
            
            # Mouse control
            "cursor_smoothing": self.cursor_smoothing,
            "movement_multiplier": self.movement_multiplier,
            "dead_zone": self.dead_zone,
            "edge_margin": self.edge_margin,
            
            # Camera
            "cam_width": self.cam_width,
            "cam_height": self.cam_height,
            
            # Visual
            "show_face_mesh": self.show_face_mesh,
            "show_eye_markers": self.show_eye_markers,
            "show_mouth_markers": self.show_mouth_markers,
            "show_stats": self.show_stats,
            "show_debug_values": self.show_debug_values,
            
            # Metadata
            "calibration_date": self.calibration_date,
            "calibration_version": self.calibration_version,
            "eye_open_range": self.eye_open_range,
            "eye_closed_range": self.eye_closed_range,
            "mouth_closed_range": self.mouth_closed_range,
            "mouth_open_range": self.mouth_open_range
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Calibrated configuration saved to: {path}")
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("CALIBRATED CONFIGURATION SUMMARY")
        print("="*70)
        print(f"\n📅 Calibration Date: {self.calibration_date}")
        print(f"\n👁️  EYE DETECTION:")
        print(f"   Closure threshold:     {self.eye_closure_thresh:.3f}")
        print(f"   Your eye open range:   {self.eye_open_range[0]:.3f} - {self.eye_open_range[1]:.3f}")
        print(f"   Your eye closed range: {self.eye_closed_range[0]:.3f} - {self.eye_closed_range[1]:.3f}")
        print(f"   Wink difference:       {self.wink_ar_diff_thresh:.3f}")
        
        print(f"\n👄 MOUTH DETECTION:")
        print(f"   Opening threshold:     {self.mouth_open_thresh:.3f}")
        print(f"   Your closed range:     {self.mouth_closed_range[0]:.3f} - {self.mouth_closed_range[1]:.3f}")
        print(f"   Your open range:       {self.mouth_open_range[0]:.3f} - {self.mouth_open_range[1]:.3f}")
        
        print("\n" + "="*70)
        print("These values are optimized for YOUR face!")
        print("="*70 + "\n")


# Create and save the calibrated configuration
if __name__ == "__main__":
    config = CalibratedConfig()
    config.print_summary()
    config.save()
    
    print("\n📋 NEXT STEPS:")
    print("1. The file 'mouse_config.json' has been created")
    print("2. Your main virtual mouse script will automatically load these values")
    print("3. If detection still needs tweaking, you can adjust:")
    print("   - Press +/- during runtime to adjust mouth sensitivity")
    print("   - Press 'D' to show debug values in real-time")
    print("   - Re-run calibration if needed")
    print("\n✓ Configuration ready!")