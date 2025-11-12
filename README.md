# virtual-mouse-using-nose-and-eye-movement

install Miniconda to download complex C packages like dlib from drive link : (https://drive.google.com/file/d/1kT2CeuDSm7HL4M9swc4pMuRHg-AyMGU9/view?usp=drive_link)[miniconda] 
Open the .exe file you just downloaded.

Click "Next" on the welcome screen.

Agree to the license agreement.

For "Install for:", choose "Just Me" (this is the recommended and easiest option).

Choose a destination folder (the default is usually fine).

Advanced Options: This is important.

Do NOT check "Add Miniconda3 to my PATH environment variable." (This can conflict with other Python installs).

DO check "Register Miniconda3 as my default Python 3.10" (or whichever version it shows).

Click "Install" and let it finish.

### After installation run following command on anaconda prompt : 

#### Create a New conda Environment: This creates a clean, isolated space for your project. Let's call it headmouse-env.

> conda create -n headmouse-env python=3.10

It will ask you to proceed (y/n). Type y and press Enter.

#### Activate Your New Environment:

> conda activate headmouse-env

#### Install dlib (The Easy Way): Install it from the "conda-forge" channel, which has pre-built versions.

> conda install -c conda-forge dlib=19.24.0

after these steps continue with downloading packages from your requirements.txt file on your code terminal :

> pip install -r requirements.txt 


This HCI (Human-Computer Interaction) application in Python(3.6) will allow you to control your mouse cursor with your facial movements, works with just your regular webcam. Its hands-free, no wearable hardware or sensors needed.

The list of actions include:

>Squinting your eyes (squint - To look with the eyes partly closed, as in bright sunlight)

>Winking for left and right click

>Moving your head around (pitch and yaw) for cursor movement

>Opening your mouth to activate the virtual mouse

#Code requirements
>Numpy - 1.13.3

>OpenCV - 3.2.0

>PyAutoGUI - 0.9.36

>Dlib - 19.4.0

>Imutils - 0.4.6

#Execution

Order of Execution is as follows:


Follow these installation guides -

https://pypi.org/project/numpy/,
https://pypi.org/project/numpy/, 
https://pyautogui.readthedocs.io/en/latest/install.html, https://pyautogui.readthedocs.io/en/latest/install.html,
https://github.com/jrosebr1/imutils 

and install the right versions of the libraries (mentioned above).

Make sure you have the model downloaded. Read the README.txt file inside the model folder for the link.

python mouse-cursor-control.py

Please raise an issue in case of any errors.

#Usage
![githubmyproject](https://github.com/g-priyanshu/virtual-mouse-using-nose-and-eye-movement/assets/134190092/d8bc8ec8-a4e5-44af-8aaf-be53d1fdaf2a)

#How it works

Dlib's prebuilt model, helps us to find the 68 facial landsmarks of a face using which a Eye aspect ratio and mouth aspect ratio is calculted and these values are then used to trigger the certain mouse actiona using PyautoGUI





