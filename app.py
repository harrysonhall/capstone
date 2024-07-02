import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

app = Flask(__name__, static_folder='src')
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://<your-public-ip>:5000"])

# Function to detect objects in an image
def detect_objects(image):
    # Perform inference
    results = model(image)

    # Results in pandas DataFrame format
    df = results.pandas().xyxy[0]

    # Extract the names of the detected objects
    detected_objects = df['name'].tolist()

    return detected_objects, df

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, df):
    for _, row in df.iterrows():
        xmin, ymin, xmax, ymax, label = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name']
        confidence = row['confidence']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Function to save image to base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

# Function to generate and return plots as base64
def generate_confidence_distribution(df):
    plt.figure(figsize=(8, 8))
    sns.barplot(x=df.index, y='confidence', hue='name', data=df, dodge=False, errorbar=None, palette='bright')
    plt.xlabel('Detection Index')
    plt.ylabel('Confidence')
    plt.title('Confidence Scores of Detected Objects')
    plt.legend(title='Object Type')
    plt.ylim(0, 1)
    plt.grid(False)
    buffer = BytesIO()
    plt.savefig(buffer, format='jpeg')
    plt.close()
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

def generate_class_distribution(df):
    plt.figure(figsize=(8, 8))
    class_counts = df['name'].value_counts()
    class_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('bright'))
    plt.title('Class Distribution of Detected Objects')
    plt.ylabel('')
    buffer = BytesIO()
    plt.savefig(buffer, format='jpeg')
    plt.close()
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

# Define the detection endpoint
@app.route('/detect', methods=['POST'])
def detect():
    if 'images' not in request.files:
        return jsonify([])

    files = request.files.getlist('images')
    if not files:
        return jsonify([])

    results = []
    for file in files:
        # Read image from the request
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect objects in the image
        detected_objects, df = detect_objects(image)

        # Generate processed images as base64
        image_with_boxes = draw_bounding_boxes(image.copy(), df)
        image_with_boxes_base64 = image_to_base64(image_with_boxes)
        confidence_distribution_base64 = generate_confidence_distribution(df)
        class_distribution_base64 = generate_class_distribution(df)

        # Add the result for this image to the results list
        results.append({
            'file_name': file.filename,
            'detected_objects': detected_objects,
            'image_with_boxes': image_with_boxes_base64,
            'confidence_distribution': confidence_distribution_base64,
            'class_distribution': class_distribution_base64
        })

    # Return the detected objects
    return jsonify(results)

# Define the hello world endpoint
@app.route('/hello', methods=['GET'])
def hello():
    return "Hello, World!"

# Serve the index.html at the root endpoint
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Serve static files (CSS, JS)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
