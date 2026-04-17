import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, redirect, url_for
import tempfile
import os

# Flask application setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load YOLOv8 models with error handling
try:
    garbage_model = YOLO('last.pt')  # Garbage detection model
    print("✓ Garbage detection model loaded successfully")
except Exception as e:
    print(f"Error loading garbage detection model: {e}")
    garbage_model = None

try:
    crowd_model = YOLO('yolov8s.pt')  # Crowd detection model
    print("✓ Crowd detection model loaded successfully")
except Exception as e:
    print(f"Error loading crowd detection model: {e}")
    crowd_model = None

try:
    violence_model = YOLO('best.pt')  # Violence detection model
    print("✓ Violence detection model loaded successfully")
except Exception as e:
    print(f"Error loading violence detection model: {e}")
    violence_model = None

# Dictionary to track active video streams
active_streams = {}

# Function to process YOLOv8 Crowd detection
def detect_crowd_yolov8(frame, stream_id=None):
    if crowd_model is None:
        return frame, 0
    
    # Run YOLOv8 crowd detection
    results = crowd_model(frame)
    
    # Count persons detected (class ID 0 in COCO dataset is 'person')
    person_count = 0
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id == 0:  # Person class
            person_count += 1
    
    # Annotate frame with YOLOv8 results
    annotated_frame = results[0].plot()
    
    # Display crowd count and alert
    cv2.putText(annotated_frame, f"Crowd size: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if person_count > 10:
        cv2.putText(annotated_frame, "Crowd size exceeded!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return annotated_frame, person_count

# Define a generator function to stream video frames to the web page
def generate(file_path, stream_id, detection_type='garbage'):
    print(f"File Path: {file_path}")
    print(f"Detection Type: {detection_type}")
    
    # Initialize stream as active
    active_streams[stream_id] = True
    
    # Initialize statistics for this stream
    detection_stats[stream_id] = {
        'garbage_count': 0,
        'violence_count': 0,
        'crowd_count': 0,
        'frames_processed': 0,
        'total_process_time': 0,
        'avg_process_time': 0,
        'inference_speed': 0,
        'detection_rate': 0
    }
    
    # Initialize logs for this stream
    detection_logs[stream_id] = []
    
    def add_log(log_type, message, details=None):
        """Add a log entry"""
        from datetime import datetime
        log_entry = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': log_type,
            'message': message,
            'details': details
        }
        detection_logs[stream_id].append(log_entry)
        # Keep only last 100 logs
        if len(detection_logs[stream_id]) > 100:
            detection_logs[stream_id].pop(0)
    
    import time
    
    add_log('system', 'Detection started', f'Source: {file_path}')
    
    # Check if file is an image
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    is_image = file_path.lower().endswith(image_extensions)
    
    if is_image:
        # Handle image file
        frame = cv2.imread(file_path)
        if frame is None:
            print("Error: Could not read image file.")
            active_streams[stream_id] = False
            return
        
        start_time = time.time()
        
        # Run detection based on selected type
        combined_frame = frame
        garbage_count = 0
        violence_count = 0
        crowd_count = 0
        
        if detection_type == 'garbage' and garbage_model:
            garbage_results = garbage_model(frame)
            combined_frame = garbage_results[0].plot()
            garbage_count = len(garbage_results[0].boxes)
        elif detection_type == 'violence' and violence_model:
            violence_results = violence_model(frame)
            combined_frame = violence_results[0].plot()
            violence_count = len(violence_results[0].boxes)
        elif detection_type == 'crowd' and crowd_model:
            combined_frame, crowd_count = detect_crowd_yolov8(frame, stream_id)
        
        # Update statistics
        process_time = (time.time() - start_time) * 1000  # Convert to ms
        detection_stats[stream_id]['frames_processed'] = 1
        detection_stats[stream_id]['avg_process_time'] = round(process_time, 2)
        
        # Update detection statistics
        detection_stats[stream_id]['garbage_count'] = garbage_count
        detection_stats[stream_id]['violence_count'] = violence_count
        detection_stats[stream_id]['crowd_count'] = crowd_count
        
        # Add logs
        if garbage_count > 0:
            add_log('garbage', f'Detected {garbage_count} garbage item(s)', 
                   f'{process_time:.1f}ms processing time')
        if violence_count > 0:
            add_log('violence', f'Detected {violence_count} violence event(s)', 
                   f'{process_time:.1f}ms processing time')
        if crowd_count > 0:
            add_log('crowd', f'Crowd size {crowd_count} detected', 
                   f'Alert: {"Exceeded" if crowd_count > 10 else "Normal"}')
        
        add_log('success', 'Image processed successfully', 
               f'Total time: {process_time:.1f}ms')

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', combined_frame)

        if ret:
            # Yield the JPEG data to Flask (for image, we just send it once)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            print("Error: Could not encode frame to JPEG.")
        
        active_streams[stream_id] = False
        return
    
    # Handle video file
    if file_path == "camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Could not open video file or camera.")
        active_streams[stream_id] = False
        return

    while cap.isOpened() and active_streams.get(stream_id, False):
        frame_start_time = time.time()
        
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # Check if stream should stop
        if not active_streams.get(stream_id, False):
            print("Stream stopped by user.")
            break

        # Run detection based on selected type
        garbage_in_frame = 0
        violence_in_frame = 0
        crowd_count = 0
        
        if detection_type == 'garbage' and garbage_model:
            garbage_results = garbage_model(frame)
            combined_frame = garbage_results[0].plot()
            garbage_in_frame = len(garbage_results[0].boxes)
        elif detection_type == 'violence' and violence_model:
            violence_results = violence_model(frame)
            combined_frame = violence_results[0].plot()
            violence_in_frame = len(violence_results[0].boxes)
        elif detection_type == 'crowd' and crowd_model:
            combined_frame, crowd_count = detect_crowd_yolov8(frame, stream_id)
        else:
            combined_frame = frame
        
        # Update statistics
        frame_process_time = (time.time() - frame_start_time) * 1000  # Convert to ms
        detection_stats[stream_id]['frames_processed'] += 1
        detection_stats[stream_id]['total_process_time'] += frame_process_time
        detection_stats[stream_id]['avg_process_time'] = round(
            detection_stats[stream_id]['total_process_time'] / detection_stats[stream_id]['frames_processed'], 2
        )
        
        # Update detection counts
        current_frames = detection_stats[stream_id]['frames_processed']
        
        if garbage_in_frame > 0:
            detection_stats[stream_id]['garbage_count'] += garbage_in_frame
            add_log('garbage', f'Frame {current_frames}: {garbage_in_frame} garbage item(s) detected', 
                   f'{frame_process_time:.1f}ms')
        if violence_in_frame > 0:
            detection_stats[stream_id]['violence_count'] += violence_in_frame
            add_log('violence', f'Frame {current_frames}: {violence_in_frame} violence event(s) detected', 
                   f'{frame_process_time:.1f}ms')
        if crowd_count > 0:
            add_log('crowd', f'Frame {current_frames}: Crowd size {crowd_count} detected', 
                   f'Alert: {"Exceeded" if crowd_count > 10 else "Normal"}')
        
        # Log frame processing every 10 frames
        if current_frames % 10 == 0:
            add_log('system', f'Processed {current_frames} frames', 
                   f'Avg: {detection_stats[stream_id]["avg_process_time"]:.1f}ms/frame')
        
        # Calculate detection rate
        detections = detection_stats[stream_id]['garbage_count'] + detection_stats[stream_id]['violence_count']
        detection_stats[stream_id]['detection_rate'] = round((detections / current_frames) * 100, 1) if current_frames > 0 else 0
        detection_stats[stream_id]['inference_speed'] = round(1000 / frame_process_time, 1) if frame_process_time > 0 else 0

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', combined_frame)

        if not ret:
            print("Error: Could not encode frame to JPEG.")
            break

        # Yield the JPEG data to Flask
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        if cv2.waitKey(1) == 27:  # Exit when ESC key is pressed
            break

    cap.release()
    active_streams[stream_id] = False
    print(f"Stream {stream_id} ended gracefully.")

# Define a route to serve the video stream
@app.route('/video_feed')
def video_feed():
    file_path = request.args.get('file')
    stream_id = request.args.get('stream_id', 'default')
    detection_type = request.args.get('detection_type', 'garbage')
    return Response(generate(file_path, stream_id, detection_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route to serve the HTML page with the file upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    global terminate_flag
    if request.method == 'POST':
        # Get detection type and upload type
        detection_type = request.form.get('detection_type', 'garbage')
        upload_type = request.form.get('upload_type', 'image')
        
        # Handle camera input
        if upload_type == 'camera':
            return redirect(url_for('results', 
                                  file='camera', 
                                  detection_type=detection_type,
                                  upload_type='camera',
                                  filename='Live Camera'))
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                
                # Redirect to results page with file info
                return redirect(url_for('results', 
                                      file=file_path, 
                                      detection_type=detection_type,
                                      upload_type=upload_type,
                                      filename=file.filename))
            else:
                return render_template('index.html', message="Please select a file to upload.")
        else:
            return render_template('index.html', message="Please select a file to upload.")
    return render_template('index.html')

# Route for displaying results
@app.route('/results')
def results():
    file_path = request.args.get('file')
    detection_type = request.args.get('detection_type', 'garbage')
    upload_type = request.args.get('upload_type', 'image')
    filename = request.args.get('filename', 'Unknown')
    
    # Generate unique stream ID
    import time
    stream_id = f"stream_{int(time.time() * 1000)}"
    
    # Determine media type from file extension or camera
    if file_path == 'camera':
        media_type = 'camera'
    else:
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        media_type = 'image' if file_path and file_path.lower().endswith(image_extensions) else 'video'
    
    return render_template('results.html', 
                          file_path=file_path,
                          detection_type=detection_type,
                          media_type=media_type,
                          file_name=filename,
                          stream_id=stream_id)

@app.route('/stop', methods=['POST'])
def stop():
    # Get stream ID from request
    stream_id = request.form.get('stream_id', 'default')
    
    # Stop the specific stream
    if stream_id in active_streams:
        active_streams[stream_id] = False
        return f"Stream {stream_id} has been stopped"
    else:
        # Stop all active streams
        for sid in list(active_streams.keys()):
            active_streams[sid] = False
        return "All streams have been stopped"

# Store detection statistics per stream
detection_stats = {}

# Store detection logs per stream
detection_logs = {}

@app.route('/stats/<stream_id>')
def get_stats(stream_id):
    """API endpoint to get detection statistics for a stream"""
    import json
    stats = detection_stats.get(stream_id, {
        'garbage_count': 0,
        'violence_count': 0,
        'crowd_count': 0,
        'frames_processed': 0,
        'avg_process_time': 0,
        'inference_speed': 0,
        'detection_rate': 0
    })
    return json.dumps(stats)

@app.route('/logs/<stream_id>')
def get_logs(stream_id):
    """API endpoint to get detection logs for a stream"""
    import json
    logs = detection_logs.get(stream_id, [])
    return json.dumps(logs)

if __name__ == '__main__':
    app.run(debug=True)
