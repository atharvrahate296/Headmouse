# Virtual Mouse – Hand & Head Gesture Based HCI System

## Overview

This project implements an advanced **Human–Computer Interaction (HCI) Virtual Mouse system** that allows users to control the mouse cursor **without any physical device**, using only a standard webcam.

The system supports **two independent control modes**:

* **Hand Mouse** – gesture-based mouse control using hand landmarks
* **Head Mouse** – face, eye, mouth, and nose–based mouse control

Both modes are integrated into a **single desktop GUI application**, making it easy to launch, switch, and use.

---

## Key Features

### 🖐️ Hand Mouse

* Relative (velocity-based) cursor movement using wrist position
* Dead-zone based stabilization for precise control
* Gesture-based actions:

  * Index + Thumb pinch → Left Click
  * Middle + Thumb pinch → Right Click
  * Peace sign → Toggle Scroll Mode
  * Fist → Cursor Lock / Unlock
  * Thumb + Pinky → Volume Control

### 👤 Head Mouse

* Relative nose-based cursor control (velocity-based)
* Mouth-open detection to activate cursor movement
* Eye gestures:

  * Left Wink → Left Click
  * Right Wink → Right Click
  * Blink → Toggle Scroll Mode
* Automatic recalibration when mouth closes
* Optional absolute iris-tracking mode

### 🖥️ GUI Application

* Single launcher for Hand Mouse and Head Mouse
* Detailed in-app usage guides
* Picture-in-Picture (PIP) camera mode
* Visual feedback (control circles, speed indicators)

---

## Project Structure

```
Virtual-Mouse/
│
├── calibrator/
│   ├── calibratedconfig.py
│   ├── calibrator.py
│   └── calibration_data_*.json
│
├── docs/
│   ├── index.html
│   └── Snapshot.png
│   └── Virtual-Mouse.exe
│
├── hand_mouse.py        
├── head_mouse.py        
├── main.py              
├── mouse_config.json    
├── requirements.txt     
├── project_setup.md
└── README.md
```

---

## Execution Flow

1. **main.py** launches the GUI
2. User selects:

   * Hand Mouse → runs `hand_mouse.py`
   * Head Mouse → runs `head_mouse.py`
3. MediaPipe processes webcam input
4. Gestures are detected and mapped to mouse actions
5. PyAutoGUI performs system-level cursor actions

---
## 🐍 Miniconda Installation Guide

To install complex C packages (like dlib) easily, we use Miniconda.

- Download Miniconda from this Drive link:  [MiniConda]([https://drive.google.com/file/d/1kT2CeuDSm7HL4M9swc4pMuRHg-AyMGU9/view?usp=drive_link](https://drive.google.com/drive/folders/16vmWMv_jmJE0ddzHCuMxrLdpCELP6y2M?usp=sharing)) 

- Open the .exe file you just downloaded.

- Click "Next" on the welcome screen.

- Agree to the license agreement.

- For "Install for:", choose "Just Me" (this is the recommended and easiest option).

- Choose a destination folder (the default is usually fine).

###  Advanced Options (⚠️ Important):

- ❌ Do NOT check "Add Miniconda3 to my PATH environment variable." (This can conflict with other Python installs).

- ✅ DO check "Register Miniconda3 as my default Python 3.10" (or whichever version it shows).

- Click "Install" and let it finish.

## Running the Project (Source Code)

### Create Conda Environment

> conda create -n headmouse-env python=3.10.19

### Activate Conda Environment

> conda activate headmouse-env

### Install Dependencies

> pip install -r requirements.txt

### Run Application

> python main.py

---

## Executable (.exe) Version

For ease of use and deployment, the project has also been **packaged into a standalone Windows executable using PyInstaller**.

### Benefits

* No need to install Python or dependencies
* One-click execution
* Suitable for demonstrations and end users

### Build Command (used during development)

```
pyinstaller --noconfirm --onefile --windowed --collect-all mediapipe --collect-all cv2 --collect-all jax --name "Virtual-Mouse" main.py
```

The generated executable is available inside the `dist/` directory after build.

---

## System Requirements

* OS: Windows 10 / 11
* Camera: Any standard webcam
* RAM: Minimum 4 GB
* Python: 3.10.19 (for source-based execution)

---

## Technologies Used

* Python 3.10.19
* OpenCV
* MediaPipe (Hands & Face Mesh)
* PyAutoGUI
* NumPy
* Tkinter (GUI)
* PIL (Image handling)
* PyInstaller (Executable packaging)

---



## How It Works (Conceptual)

* Webcam feed is processed in real time
* MediaPipe extracts hand or facial landmarks
* Relative displacement from a reference point is calculated
* Velocity-based cursor movement is generated
* Gestures trigger mouse clicks, scrolling, and other actions

---

## Use Case

* Accessibility solutions
* Hands-free computer interaction
* Assistive technology
* HCI research & academic projects

---

## Support

If you face any issues during setup or execution, please raise an issue or contact the project maintainer.

---

