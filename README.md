RailGuard-AI

AI-Based Real-Time Surveillance and Threat Detection System

⸻

Overview

RailGuard-AI is an intelligent surveillance and threat detection system designed to identify violence, suspicious activities, abandoned objects, and potential security threats in real time using deep learning and computer vision.

The system leverages YOLOv8-based object detection along with custom-trained AI models to continuously monitor live CCTV/video feeds and provide instant analysis for enhanced public safety. It is highly suitable for railway stations, airports, smart cities, malls, and other high-traffic public environments.

The project aims to improve automated surveillance by reducing manual monitoring efforts and enabling faster threat response through AI-powered analytics.

⸻

Features

Core Features

* Real-time object detection using YOLOv8
* Violence and non-violence classification
* Suspicious activity detection
* Abandoned object detection
* Live CCTV/video stream processing
* Custom-trained deep learning models
* Flask-based web dashboard
* Multi-object tracking support
* Real-time monitoring and analysis
* Scalable smart surveillance architecture

Advanced Features

* High accuracy detection with optimized inference
* Supports webcam, CCTV, and uploaded video feeds
* Modular architecture for easy feature expansion
* Lightweight deployment capability
* Automated threat identification pipeline
* Low-latency frame processing
* Future-ready cloud integration support
* Dataset training and fine-tuning support
* Easy integration with existing surveillance infrastructure
* Smart alert generation framework

⸻

Tech Stack

AI / Machine Learning

* PyTorch
* YOLOv8 (Ultralytics)
* Scikit-learn
* TensorFlow (optional future integration)

Computer Vision

* OpenCV

Backend

* Flask
* Flask-RESTful

Data Processing & Visualization

* NumPy
* Pandas
* Matplotlib
* Seaborn

Deployment & Tools

* Git & GitHub
* Jupyter Notebook
* Google Colab
* VS Code

⸻

System Workflow

1. Video feed is captured from CCTV/Webcam
2. Frames are processed in real time
3. YOLOv8 detects humans and suspicious objects
4. Violence detection model classifies activities
5. Threat analysis module evaluates risk level
6. Results are displayed on the Flask dashboard
7. Future alert system can trigger notifications

⸻

Project Structure

RailGuard-AI/
│── app.py                  # Flask web application
│── detection_system.py     # Core detection logic
│── garbage_detection.py    # Additional detection module
│── models/
│   ├── best.pt             # Trained model
│   ├── last.pt             # Latest checkpoint
│   ├── yolov8s.pt          # Pretrained model
│── notebooks/
│   ├── train_yolov8.ipynb
│   ├── violence_detection.ipynb
│── config/
│   ├── yolov3.cfg
│   ├── coco.names
│── static/                 # CSS, JS, assets
│── templates/              # HTML templates
│── requirements.txt
│── README.md

⸻

Installation

1. Clone the Repository

git clone https://github.com/your-username/railguard-ai.git
cd railguard-ai

2. Install Dependencies

pip install -r requirements.txt

⸻

How to Run

Run Flask Application

python app.py

Run Detection System

python detection_system.py

⸻

Models Used

* YOLOv8 pretrained model: yolov8s.pt
* Custom-trained detection models: best.pt, last.pt
* Violence detection classifier
* Real-time object recognition models

⸻

Use Cases

* Railway station surveillance
* Smart city monitoring
* Airport security systems
* Public safety surveillance
* Crowd monitoring systems
* Suspicious activity detection
* CCTV-based automated threat analysis
* Metro station monitoring
* Shopping mall security
* Traffic and public area monitoring

⸻

Performance Highlights

* Real-time inference capability
* Optimized object detection pipeline
* Supports GPU acceleration
* Efficient frame-by-frame analysis
* Capable of detecting multiple objects simultaneously

⸻

Limitations

* Requires GPU for optimal performance
* Detection accuracy depends on dataset quality
* Performance may vary in low-light conditions
* Real-time processing speed depends on hardware specifications
* False positives may occur in crowded environments

⸻

Future Improvements

* SMS/Email alert system
* Cloud deployment using AWS/GCP/Azure
* Mobile application integration
* Face recognition integration
* AI-powered anomaly detection
* Real-time incident reporting dashboard
* Audio-based threat detection
* Edge AI deployment for IoT devices
* Multi-camera synchronization
* Enhanced activity classification models

⸻

Research & Innovation Scope

* Smart AI-based public surveillance
* Edge computing for real-time security
* Deep learning for violence detection
* Automated public safety analytics
* AI-driven smart transportation security systems

⸻

Author

Aniruddha Sutawane
GitHub: https://github.com/aniruddha07s
