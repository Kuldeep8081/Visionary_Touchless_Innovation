👁️ Visionary Touchless Innovation

This repository contains a collection of Python scripts dedicated to enabling touchless and voice-controlled interaction with a computer system. By leveraging computer vision, facial recognition, and voice commands, this project aims to create a more intuitive, hygienic, and accessible user experience.

✨ Features

The project is built on modular Python scripts, each focused on a specific touchless control capability:
Script,Description
Mouse_control.py,Allows full control of the mouse cursor and clicks using hand gestures captured by the webcam.
face_lock.py,"Implements a simple Face Lock system for computer security, automatically locking the screen when the authorized user moves away."
voice_control.py,Enables key system actions and application launching through predefined Voice Commands.
screenshot_through_eye_blinking.py,"Allows users to capture a screenshot simply by blinking their eyes, providing a quick and completely hands-free method."
keyboard.py,"Provides a foundation for controlling keyboard inputs through non-traditional means (e.g., gestures or voice)."
B_Vcontrol.py,Enables touchless control over system functions like Screen Brightness and Volume using hand or face movements.

💻 Technology Stack

Language: Python (100%)

Core Libraries: Likely uses established Computer Vision libraries such as:

OpenCV (for video and image processing)

MediaPipe (for hand/face landmark detection)

pyautogui (for system control/virtual mouse/keyboard)

dlib (for facial recognition, if used in face_lock.py)

💻 Technology Stack

Language: Python (100%)

Core Libraries: Likely uses established Computer Vision libraries such as:

OpenCV (for video and image processing)

MediaPipe (for hand/face landmark detection)

pyautogui (for system control/virtual mouse/keyboard)

dlib (for facial recognition, if used in face_lock.py)

git clone https://github.com/Kuldeep8081/Visionary_Touchless_Innovation.git
cd Visionary_Touchless_Innovation

3. Install Dependencies
4. 
You will need to install the necessary Python libraries. While the exact list is dependent on the internal code, the following command should cover the likely major packages:

# This is a suggested installation command
pip install opencv-python mediapipe pyautogui dlib numpy

▶️ Usage
To run any of the touchless modules, simply execute the respective Python file:
# Example: Run touchless mouse control
python Mouse_control.py

# Example: Run the face lock security system
python face_lock.py

# Example: Run the eye-blinking screenshot feature
python screenshot_through_eye_blinking.py

Note: A working webcam is required for all computer vision features.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📄 License
This project is open-sourced under the MIT License. See the LICENSE file for details.

Developed by Kuldeep
