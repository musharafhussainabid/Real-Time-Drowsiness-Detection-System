# Real-Time Drowsiness Detection Using Facial Landmarks

This project implements a real-time drowsiness detection system using eye aspect ratio (EAR) derived from facial landmarks. It leverages Dlib's facial landmark predictor and uses a webcam (via Google Colab) to analyze drowsiness in individuals based on eye closure over time.

## Description

- Uses Dlib’s 68-point facial landmark detector.
- Calculates distance between key eye landmarks to monitor blinking or prolonged eye closure.
- Detects drowsiness based on average eye distances across consecutive frames.
- Displays feedback in real time on the video frame.
- Designed to run in a Jupyter Notebook on Google Colab using the device camera.

## Features

- Facial landmark detection with Dlib
- Eye aspect ratio-based drowsiness detection
- Webcam integration in Google Colab
- Dynamic visual feedback ("Drowsiness Detected" / "No Drowsiness")

## Setup Instructions

1. **Mount in Google Colab** and run all cells of `DrowsinessDetection.ipynb`.

2. **Download shape predictor model:**

The script automatically downloads:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


You may need to extract it manually if necessary:

bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

Run the notebook:

Ensure the webcam permission is granted when prompted in Colab.

File Structure

drowsiness-detection/

├── DrowsinessDetection.ipynb        # Main notebook for detection

├── shape_predictor_68_face_landmarks.dat.bz2  # Auto-downloaded shape model

├── requirements.txt                 # Python dependencies

└── README.md

Technologies Used

Python 

OpenCV

NumPy

Dlib (for facial landmark detection)

imutils

Google Colab (camera input handling)

Usage Example
The system displays real-time webcam frames and overlays:

"Drowsiness Detected" when average eye distance falls below a threshold across multiple frames.

"No Drowsiness" when eyes are open consistently.

The system is designed for short-term testing scenarios, ideal for demonstrating fatigue monitoring mechanisms.

License
This project is licensed under the MIT License.

Acknowledgments
Dlib

iBUG 300-W Dataset
