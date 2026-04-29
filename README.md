RailGuard-AI
AI-Based Real-Time Surveillance and Threat Detection System
⸻
Overview:

RailGuard-AI is an intelligent surveillance system designed to detect violence, suspicious activities, and objects in real time using deep learning and computer vision.

The system uses YOLOv8-based object detection along with custom-trained models to monitor live video feeds and identify potential threats. It can be applied in railway stations, public places, and smart city environments.
⸻
Features
* Real-time object detection using YOLOv8
* Violence and non-violence detection
* Custom-trained deep learning model
* Live video stream processing
* Flask-based web interface
* Scalable for smart surveillance systems
⸻
Tech Stack

AI / ML
* PyTorch
* YOLOv8 (Ultralytics)
* Scikit-learn

Computer Vision
* OpenCV

Backend
* Flask
* Flask-RESTful

Data and Visualization
* NumPy
* Pandas
* Matplotlib
* Seaborn
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
│── requirements.txt
│── README.md
⸻
Installation
1. Clone the repository
    git clone https://github.com/your-username/railguard-ai.git
    cd railguard-ai
2. Install dependencies
    pip install -r requirements.txt
⸻
How to Run:
Run Flask application:
python app.py

Run detection system:
python detection_system.py
⸻
Models
* YOLOv8 pretrained model: yolov8s.pt
* Custom trained models: best.pt, last.pt
⸻
Use Cases
* Railway station surveillance
* Smart city monitoring
* Public safety systems
* CCTV-based threat detection
⸻
Limitations
* Requires GPU for better performance
* Accuracy depends on dataset quality
* Real-time speed depends on hardware
⸻
Future Improvements
* Alert system (SMS or Email)
* Cloud deployment
* Mobile application integration
* Improved activity classification
⸻
Author: github.com/aniruddha07s
