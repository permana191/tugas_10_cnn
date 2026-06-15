import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

print("Memuat Model AI & Haar Cascade...")
try:
    model = load_model('model.h5') 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print("Sistem Siap!")
except Exception as e:
    print(f"Error saat memuat model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_live', methods=['POST'])
def predict_live():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image'})

    try:
        img_data = base64.b64decode(data['image'].split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        results = []

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
    
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
