from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict

app = Flask(__name__)

# Load your pre-trained model
MODEL_PATH = r"C:/Users/Jefferson/Desktop/PROJECTS/Emotion_Detection/emotional/emotion_detection_model_v2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotions list
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Anxiety', 'Embarrassed']

# Load face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Store emotion history for statistics
emotion_history = []
emotion_counts = defaultdict(int)
timestamp_emotions = []


def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    if face_img.shape[-1] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    if len(face_img.shape) == 3:
        face_img = np.expand_dims(face_img, axis=-1)
    return face_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    results = []
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        processed_face = preprocess_face(face_img)
        prediction = model.predict(processed_face)
        emotion_idx = np.argmax(prediction[0])
        emotion = EMOTIONS[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        current_time = datetime.now()
        emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': current_time
        })
        emotion_counts[emotion] += 1
        timestamp_emotions.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': current_time
        })
        if len(emotion_history) > 100:
            emotion_history.pop(0)
        if len(timestamp_emotions) > 50:
            timestamp_emotions.pop(0)
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'box': [int(x), int(y), int(w), int(h)]
        })
    return jsonify({'results': results, 'count': len(results)})

@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    file = request.files['frame']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    results = []
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        processed_face = preprocess_face(face_img)
        prediction = model.predict(processed_face)
        emotion_idx = np.argmax(prediction[0])
        emotion = EMOTIONS[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        current_time = datetime.now()
        emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': current_time
        })
        emotion_counts[emotion] += 1
        timestamp_emotions.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': current_time
        })
        if len(emotion_history) > 100:
            emotion_history.pop(0)
        if len(timestamp_emotions) > 50:
            timestamp_emotions.pop(0)
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'box': [int(x), int(y), int(w), int(h)]
        })
    return jsonify({'results': results, 'count': len(results)})

@app.route('/generate_profile', methods=['POST'])
def generate_profile():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    custom_emotion_history = data.get('emotionHistory', emotion_history)
    template = match_static_template(custom_emotion_history)
    # Calculate statistics
    counts = defaultdict(int)
    for entry in custom_emotion_history:
        counts[entry['emotion']] += 1
    total = sum(counts.values())
    dominant_count = counts.get(template['dominant'], 0)
    secondary_count = counts.get(template['secondary'], 0) if template.get('secondary') else 0
    profile_text = f"""<b>{template['title']}</b><br><br>{template['content']}<br><br><b>Your Statistics:</b><br>
    - Total detections: {total}<br>
    - Dominant emotion: {template['dominant']} ({dominant_count} occurrences)<br>
    - Secondary emotion: {template.get('secondary') or 'None'} ({secondary_count} occurrences)"""
    return jsonify({'profile': profile_text})

@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    total_emotions = sum(emotion_counts.values())
    if total_emotions == 0:
        return jsonify({'error': 'No emotion data available'}), 404
    emotion_distribution = {
        emotion: (count / total_emotions) * 100
        for emotion, count in emotion_counts.items()
    }
    timeline_chart = generate_emotion_timeline()
    return jsonify({
        'emotion_counts': dict(emotion_counts),
        'emotion_distribution': emotion_distribution,
        'total_detections': total_emotions,
        'timeline_chart': timeline_chart
    })

def generate_emotion_timeline():
    if len(timestamp_emotions) < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    timestamps = [entry['timestamp'] for entry in timestamp_emotions]
    emotions = [entry['emotion'] for entry in timestamp_emotions]
    confidences = [entry['confidence'] * 100 for entry in timestamp_emotions]
    emotion_colors = {
        'Angry': '#e74c3c', 'Disgust': '#8e44ad', 'Fear': '#9b59b6', 'Happy': '#f1c40f',
        'Sad': '#3498db', 'Surprise': '#e67e22', 'Neutral': '#95a5a6', 'Anxiety': '#1abc9c', 'Embarrassed': '#e84393'
    }
    colors = [emotion_colors.get(emotion, '#333333') for emotion in emotions]
    ax.scatter(range(len(timestamps)), confidences, c=colors, s=50)
    ax.plot(range(len(timestamps)), confidences, 'k-', alpha=0.3)
    ax.set_xlabel('Time Sequence')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Emotion Detection Timeline')
    ax.set_xticks(range(len(timestamps)))
    ax.set_xticklabels([t.strftime('%H:%M:%S') for t in timestamps], rotation=45)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=10, label=emotion)
                      for emotion, color in emotion_colors.items() if emotion in emotions]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

if __name__ == '__main__':
    app.run(debug=True)
