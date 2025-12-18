# Virtual Mouse Project - Setup Guide

## 📁 Project Structure

```
virtual-mouse/
│
├── main.py                 # Main GUI application (NEW)
├── hand_mouse.py          # Hand gesture control (MODIFIED)
├── head_mouse.py          # Head/facial control (EXISTING)
├── requirements.txt       # Cleaned dependencies (MODIFIED)
└── mouse_config.json      # Auto-generated config (created by head_mouse.py)
```

## 🔧 Changes Made

### 1. **requirements.txt** (Cleaned)
- **Removed**: 90+ unnecessary packages
- **Kept**: Only essential dependencies:
  - `opencv-python` - Camera and image processing
  - `mediapipe` - Hand and face detection
  - `numpy` - Numerical operations
  - `pyautogui` - Mouse control
  - `pillow` - Optional image support

### 2. **hand_mouse.py** (Modified)
**Changes:**
- ✅ Added `handmouse()` function (line ~500) - wrapper for GUI integration
- ✅ Modified `if __name__ == "__main__"` block to call `handmouse()`
- ✅ No breaking changes to existing functionality
- ✅ Can still be run standalone: `python hand_mouse.py`

### 3. **head_mouse.py** (No changes needed)
- ✅ Already has `headmouse()` function
- ✅ Ready for GUI integration
- ✅ Can still be run standalone: `python head_mouse.py`

### 4. **main.py** (New File)
**Features:**
- 🎨 Modern, professional GUI with dark theme
- 🎯 Two large, clickable buttons for each mode
- ✅ Automatic module availability detection
- 🔄 Thread-based execution (non-blocking)
- 📊 Status indicator showing system state
- ⚠️ Error handling and user feedback
- 🎨 Hover effects on buttons
- 📱 Responsive 800x600 window

## 🚀 Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Camera
Ensure your webcam is connected and working.

### Step 3: Run the Application
```bash
python main.py
```

## 🎮 Usage

### From GUI:
1. Launch `main.py`
2. Click **"🖐️ Hand Mouse"** or **"👤 Head Mouse"**
3. Control appears in a new window
4. Press **Q** or **ESC** in the camera window to exit

### Standalone Mode:
You can still run each module independently:
```bash
python hand_mouse.py   # Hand gesture control
python head_mouse.py   # Head/face control
```

## 🎯 Control Methods

### Hand Mouse (hand_mouse.py)
- ✋ **Open Hand** - Move cursor
- 👌 **Index+Thumb Pinch** - Left click
- 🤏 **Middle+Thumb Pinch** - Right click
- ✌️ **Peace Sign** - Toggle scroll mode
- ✊ **Fist** - Lock/unlock cursor
- 👍 **Thumbs Up** - Volume up
- 👎 **Thumbs Down** - Volume down

### Head Mouse (head_mouse.py)
- 💬 **Open Mouth** - Move cursor (nose tracking)
- 💬 **Close Mouth** - Lock cursor position
- 👁️ **Blink Both Eyes** - Toggle scroll mode
- 😉 **Wink Left** - Left click
- 😉 **Wink Right** - Right click

## 🛠️ Troubleshooting

### Camera Not Working
- Check camera permissions
- Ensure no other app is using the camera
- Try unplugging and reconnecting USB camera

### Module Not Available
- Verify all files are in the same directory
- Check file names match exactly:
  - `hand_mouse.py` (not `handmouse.py`)
  - `head_mouse.py` (not `headmouse.py`)

### Import Errors
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify Python version (3.8+ recommended)

### Performance Issues
- Close other applications using the camera
- Reduce camera resolution in code if needed
- Ensure good lighting for face/hand detection

## 📝 Notes

- **Thread Safety**: GUI runs modules in separate threads to prevent freezing
- **Failsafe**: PyAutoGUI failsafe is enabled (move mouse to corner to stop)
- **Configuration**: head_mouse.py saves settings to `mouse_config.json`
- **Compatibility**: Tested on Windows, should work on macOS/Linux

## 🎨 GUI Features

- **Status Indicator**: Shows "Ready" (green), "Active" (yellow), or errors (red)
- **Hover Effects**: Buttons highlight on mouse hover
- **Error Messages**: Clear feedback if modules are missing
- **Threading**: Non-blocking execution allows GUI to remain responsive
- **Single Instance**: Prevents multiple controls running simultaneously

## ⚡ Performance Tips

1. **Good Lighting**: Ensures better hand/face detection
2. **Stable Camera**: Mount camera securely for head mouse
3. **Clear Background**: Reduces false detections
4. **Calibration**: Use keyboard shortcuts in head_mouse.py to adjust sensitivity
5. **Close Unused Apps**: Frees up camera and system resources

---

**Ready to use!** Just run `python main.py` and start controlling your mouse with gestures! 🚀