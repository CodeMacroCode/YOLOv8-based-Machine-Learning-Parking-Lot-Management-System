from flask import Flask, render_template, Response
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

app = Flask(__name__)

ZONE_POLYGON = np.array([
    [0, 0],
    [1280, 0],
    [1280, 720],
    [0, 720]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

frame_width, frame_height = (1280, 720)  # Set your desired frame resolution here

cap = cv2.VideoCapture(0) # add 0 for web cam or 1 for external cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

model = YOLO() # Your YOLOv8 trained model

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=(frame_width, frame_height))
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red())

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        detections = detect_objects(frame)

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame with boundary for video streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_objects(frame):
    result = model(frame)[0]
    detections = sv.Detections.from_yolov8(result)
    detections = detections[detections.class_id == 1]
    return detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/run-python')
def run_python():
    # Get the current frame from the webcam
    success, frame = cap.read()
    if not success:
        return "Error: Failed to capture frame from webcam"

    # Perform object detection
    detections = detect_objects(frame)

    # Format the detection results for display
    detection_results = ""
    for detection in detections:
        detection_results += f"Class: {detection.class_name}, Confidence: {detection.confidence}<br>"

    # Add debug code
    debug_info = f"Number of Detections: {len(detections)}"

    return render_template('index.html', detection_results=detection_results, debug_info=debug_info)

if __name__ == "__main__":
    app.run(debug=True)
