🚦 Traffic Violation Detection System (TVDS)
📌 Overview

The Traffic Violation Detection System (TVDS) is a computer vision–based project developed to detect common traffic violations using images and videos captured from CCTV-style cameras.
The system is designed as a prototype to demonstrate how Artificial Intelligence and Computer Vision can assist in automated traffic monitoring.

This project focuses on rule-based violation detection combined with deep learning object detection.

🎯 Objectives of the Project

Detect traffic rule violations automatically

Reduce dependency on manual traffic monitoring

Demonstrate real-world application of computer vision

Provide a scalable prototype for smart traffic systems

🚨 Traffic Violations Detected

The system detects the following violations:

Helmet Violation

Detects riders on two-wheelers without helmets

Helmet detection is logic-based (prototype approach)

Red Light Violation

Detects vehicles crossing the stop line during red signal

Yellow Light Crossing

Detects vehicles crossing during yellow signal (not counted as violation but highlighted)

Wrong Direction Detection

Detects vehicles moving opposite to the expected traffic direction

🧠 Technologies Used
🔹 Python

Used as the main programming language to implement all system logic.

🔹 YOLOv8 (You Only Look Once)

A deep learning model used for:

Vehicle detection

Person detection

Traffic light detection

Object tracking in videos

🔹 Ultralytics Framework

Provides an efficient implementation of YOLOv8 with built-in tracking support.

🔹 OpenCV

Used for:

Image and video processing

Drawing bounding boxes

Color detection (red/yellow signal)

Displaying output

🔹 NumPy

Used for numerical operations and pixel-level calculations.

🧩 System Working
Image Mode

Processes static traffic images

Detects vehicles and riders

Checks helmet compliance

Displays violations visually

Video Mode

Processes pre-recorded traffic videos

Uses object tracking to assign unique IDs

Detects signal color (red/yellow/green)

Checks stop-line crossing

Detects wrong-direction movement

🗂️ Project Structure
TVDS_Project/
│
├── main.py           # Main program
├── config.py         # Mode selection (image / video)
├── images/           # Input images
├── videos/           # Input videos
├── .gitignore
└── README.md

▶️ How to Run the Project
1️⃣ Install Required Libraries
pip install ultralytics opencv-python numpy

2️⃣ Select Mode

In config.py:

mode = "image"   # or "video"

3️⃣ Run the Project
python main.py

📊 Output

Bounding boxes on detected objects

Labels indicating violations

Unique IDs for vehicles in video mode

Visual demonstration suitable for academic presentation

⚠️ Limitations

Helmet detection is rule-based and may not be 100% accurate

Performance depends on camera angle and lighting

Designed as a prototype, not a production system

🚀 Future Enhancements

Integrate a helmet-trained deep learning model

Add number plate recognition (ANPR)

Support real-time CCTV feeds

Improve accuracy using custom-trained datasets

🎓 Academic Purpose

This project is developed as a minor academic project to demonstrate the application of AI and Computer Vision concepts in real-world traffic management systems.

👤 Author

Vivek Dhangar
GitHub: https://github.com/vivekdhangar1903

📜 License

This project is intended for educational use only.

🔥 Tip (IMPORTANT)

After adding this README:

Save file as README.md

Run:

git add README.md
git commit -m "Added project README"
git push
