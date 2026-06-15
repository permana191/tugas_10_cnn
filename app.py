from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import uuid
import base64
import cv2  # Tambahan OpenCV untuk Auto-Crop & Real-Time Bounding Box

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Memuat Model AI & Haar Cascade...")
model = load_model('models/model_emosi.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar-Cascade dari OpenCV untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Sistem Siap!")

# FITUR 1: Smart Face Auto-Crop
def process_image_for_prediction(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah di dalam gambar
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) > 0:
        # Jika wajah ditemukan, potong (crop) hanya bagian wajahnya saja
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
    else:
        # Jika tidak ada wajah, gunakan seluruh gambar
        roi_gray = gray

    # Preprocessing untuk model CNN
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_array = roi_gray.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1) # Tambah channel grayscale
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        filepath = None
        if 'webcam_image' in request.form and request.form['webcam_image'] != '':
            image_data = request.form['webcam_image'].split(',')[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.png")
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(image_data))
                
        elif 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(f"{uuid.uuid4().hex}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
        if filepath:
            processed_image = process_image_for_prediction(filepath)
            prediction = model.predict(processed_image)[0]
            
            # FITUR 2: Menyiapkan data untuk Grafik Chart.js
            emotion_probs = [round(float(p) * 100, 2) for p in prediction]
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            return render_template('result.html', 
                                   emotion=predicted_class, 
                                   confidence=round(confidence, 2),
                                   image_path=filepath,
                                   probs=emotion_probs,
                                   labels=class_labels)
        return redirect(request.url)
    return render_template('index.html')

# FITUR 3: API Endpoint untuk Real-Time Live Video Tracking
@app.route('/predict_live', methods=['POST'])
def predict_live():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image'})

    # Decode base64 frame dari video JS
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    
    # Deteksi dan prediksi multi-wajah secara real-time
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        pred = model.predict(roi, verbose=0)[0]
        idx = np.argmax(pred)
        results.append({
            'box': [int(x), int(y), int(w), int(h)],
            'emotion': class_labels[idx],
            'confidence': round(float(pred[idx]) * 100, 2)
        })
        
    return jsonify({'faces': results})

if __name__ == '__main__':
    app.run(debug=True)
    
    # ... kode sebelumnya ...

app = Flask(__name__)

# Tambahkan ini agar Vercel bisa membaca aplikasi Flask-mu
# Vercel mencari objek bernama 'app'
if __name__ == '__main__':
    app.run()