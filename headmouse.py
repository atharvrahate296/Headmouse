import sys 

# --- 1. Handle Library Imports ---
try:
    from imutils import face_utils
    from utils import * 
    import numpy as np
    import pyautogui as pag
    import imutils
    import dlib
    import cv2
except ImportError as e:
    print(f"--- FATAL ERROR: Missing Required Library ---")
    print(f"Details: {e}")
    print("One or more required libraries (dlib, opencv, imutils, pyautogui) are not installed.")
    print("Please install them, for example: 'pip install opencv-python dlib imutils pyautogui'")
    sys.exit(1)

# --- Constants and Thresholds ---
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 15
EYE_AR_THRESH = 0.22
EYE_AR_CONSECUTIVE_FRAMES = 15
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10

# --- State Counters and Flags ---
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False 
LEFT_WINK = False 
RIGHT_WINK = False 
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)

# --- Colors for Drawing ---
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

# --- Facial Landmark Indices ---
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# --- 2. Initialize Models and Camera ---
try:
    print("Loading dlib face detector...")
    shape_predictor_path = "model/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    
    print("Loading dlib shape predictor...")
    predictor = dlib.shape_predictor(shape_predictor_path)

    print("Opening video camera...")
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Error: Could not open video camera (index 0). Is it being used by another application?")

except FileNotFoundError:
    print(f"--- FATAL ERROR: Model File Not Found ---")
    print(f"Could not find '{shape_predictor_path}'.")
    print("Please download 'shape_predictor_68_face_landmarks.dat' and place it in a 'model' folder.")
    sys.exit(1)
except (IOError, RuntimeError) as e:
    print(f"--- FATAL ERROR: Initialization Failed ---")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"--- FATAL ERROR: An unexpected error occurred on startup ---")
    print(f"Details: {e}")
    sys.exit(1)

print("Initialization successful. Starting main loop...")
cam_w = 640
cam_h = 480

# --- 3. Main Application Loop ---
try:
    while True:
        # --- 3a. Per-Frame Processing and Error Skipping ---
        # This inner try/except skips a single bad frame
        # without crashing the whole application.
        try:
            _, frame = vid.read()
            
            # Check for empty frame
            if frame is None or frame.size == 0:
                print("Warning: Empty frame received. Waiting for camera...")
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC pressed. Exiting.")
                    break
                continue
            
            # --- 3b. Image Pre-processing ---
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=cam_w, height=cam_h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"DEBUG: gray.dtype={gray.dtype}, gray.shape={gray.shape}")

            # --- 3c. Face and Landmark Detection ---
            rects = detector(frame, 0)

            # If no face is detected, skip gesture processing
            if len(rects) == 0:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC pressed. Exiting.")
                    break
                continue

            rect = rects[0]
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            mouth = shape[mStart:mEnd]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            nose = shape[nStart:nEnd]

            # --- 3d. Calculate Aspect Ratios ---
            try:
                mar = mouth_aspect_ratio(mouth)
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                diff_ear = np.abs(leftEAR - rightEAR)
            except ZeroDivisionError:
                print("Warning: Bad landmarks (division by zero). Skipping frame.")
                continue 

            nose_point = (nose[3, 0], nose[3, 1])

            # --- 3e. Draw Facial Overlays ---
            mouthHull = cv2.convexHull(mouth)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
            cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

            # --- 3f. Gesture Detection Logic ---
            
            # (Wink detection for left/right clicks)
            if diff_ear > WINK_AR_DIFF_THRESH:
                if leftEAR < rightEAR:
                    if leftEAR < WINK_AR_CLOSE_THRESH:
                        WINK_COUNTER += 1
                        if WINK_COUNTER >= WINK_CONSECUTIVE_FRAMES:
                            pag.click(button='left')
                            WINK_COUNTER = 0
                    else:
                        WINK_COUNTER = 0 
                elif leftEAR > rightEAR:
                    if rightEAR < WINK_AR_CLOSE_THRESH:
                        WINK_COUNTER += 1
                        if WINK_COUNTER >= WINK_CONSECUTIVE_FRAMES:
                            pag.click(button='right')
                            WINK_COUNTER = 0
                    else:
                        WINK_COUNTER = 0
                else:
                    WINK_COUNTER = 0
            else:
                # (Blink detection for toggling scroll mode)
                if ear <= EYE_AR_THRESH:
                    EYE_COUNTER += 1
                    if EYE_COUNTER >= EYE_AR_CONSECUTIVE_FRAMES:
                        SCROLL_MODE = not SCROLL_MODE
                        EYE_COUNTER = 0
                else:
                    EYE_COUNTER = 0
                    WINK_COUNTER = 0 

            # (Mouth open for toggling mouse movement)
            if mar > MOUTH_AR_THRESH:
                MOUTH_COUNTER += 1
                if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                    INPUT_MODE = not INPUT_MODE
                    MOUTH_COUNTER = 0
                    ANCHOR_POINT = nose_point
            else:
                MOUTH_COUNTER = 0

            # --- 3g. Mouse Control Logic ---
            if INPUT_MODE:
                cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
                x, y = ANCHOR_POINT
                nx, ny = nose_point
                w, h = 60, 35 
                
                cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
                cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

                dir = direction(nose_point, ANCHOR_POINT, w, h)
                cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
                
                drag = 18 
                if dir == 'right':
                    pag.moveRel(drag, 0)
                elif dir == 'left':
                    pag.moveRel(-drag, 0)
                elif dir == 'up':
                    if SCROLL_MODE:
                        pag.scroll(40)
                    else:
                        pag.moveRel(0, -drag)
                elif dir == 'down':
                    if SCROLL_MODE:
                        pag.scroll(-40)
                    else:
                        pag.moveRel(0, drag)

            if SCROLL_MODE:
                cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

            # --- 3h. Display Frame and Handle Exit ---
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("ESC key pressed. Exiting.")
                break 

        except cv2.error as e:
            print(f"Warning: OpenCV error on a single frame: {e}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Unexpected error processing a frame: {e}. Skipping.")
            continue

# --- Handle High-Level Program Stops ---
except pag.FailSafeException:
    print("\n--- SAFETY STOP ---")
    print("PyAutoGUI fail-safe triggered (mouse moved to a corner).")
    print("Program has been stopped to prevent runaway mouse.")
except KeyboardInterrupt:
    print("\n--- User Stop ---")
    print("Program stopped by user (Ctrl+C).")

# --- 4. Cleanup ---
finally:
    print("\nCleaning up resources...")
    vid.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")