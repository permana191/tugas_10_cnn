import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. PERSIAPAN DATASET & DATA AUGMENTATION
# ==========================================
# Pastikan path ini sesuai dengan lokasi dataset hasil ekstrakmu
train_dir = 'dataset/train'
test_dir = 'dataset/test'

print("\n--- Memuat Dataset ---")
# Augmentasi HANYA untuk data latih agar model belajar berbagai variasi wajah
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,       
    width_shift_range=0.1,   
    height_shift_range=0.1,  
    horizontal_flip=True,    
    fill_mode='nearest'
)

# Data uji HANYA dinormalisasi
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

# PENTING: shuffle=False agar urutan prediksi cocok dengan label asli untuk Confusion Matrix
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical',
    shuffle=False 
)

# Mengambil nama-nama kelas secara otomatis dari folder
class_names = list(train_generator.class_indices.keys())
print(f"Kelas yang dideteksi: {class_names}")

# ==========================================
# 2. ARSITEKTUR MODEL CNN 
# ==========================================
model = Sequential([
    # Blok Konvolusi 1
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Blok Konvolusi 2
    Conv2D(128, (5, 5), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Blok Konvolusi 3
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Blok Fully Connected Layer
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), 
    Dense(7, activation='softmax') # 7 Kelas emosi
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 3. PELATIHAN MODEL & CALLBACKS
# ==========================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

print("\n--- Memulai Proses Training ---")
history = model.fit(
    train_generator,
    epochs=30, 
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# Menyimpan Model
os.makedirs('models', exist_ok=True)
model.save('models/model_emosi.h5')
print("\nModel berhasil disimpan di models/model_emosi.h5")

# ==========================================
# 4. VISUALISASI UNTUK LAPORAN (GRAFIK & MATRIKS)
# ==========================================
print("\n--- Menyiapkan Visualisasi ---")
print("* Catatan: Tutup jendela grafik pertama agar program bisa memunculkan grafik selanjutnya.")

# A. Visualisasi Akurasi dan Loss
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(history.history['accuracy'], label='Akurasi Latih', color='blue', linewidth=2)
ax[0].plot(history.history['val_accuracy'], label='Akurasi Validasi', color='orange', linewidth=2)
ax[0].set_title('Grafik Akurasi Model CNN')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Akurasi')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(history.history['loss'], label='Loss Latih', color='red', linewidth=2)
ax[1].plot(history.history['val_loss'], label='Loss Validasi', color='green', linewidth=2)
ax[1].set_title('Grafik Loss (Kesalahan) Model CNN')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()
ax[1].grid(True)
plt.tight_layout()
plt.show()

# B. Confusion Matrix
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prediksi Model")
plt.ylabel("Label Asli")
plt.title("Confusion Matrix Klasifikasi Emosi Wajah")
plt.show()

# Cetak Laporan Presisi & Recall
print("\n--- Laporan Klasifikasi ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==========================================
# 5. VISUALISASI HASIL PREDIKSI (GAMBAR)
# ==========================================
print("\n--- Menampilkan Contoh Hasil Prediksi (Visual) ---")
# Reset generator lalu ambil 1 batch acak untuk diuji
validation_generator.reset()
validation_generator.shuffle = True
x_test_batch, y_test_batch = next(iter(validation_generator))

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    
    img = x_test_batch[i]
    true_label_idx = np.argmax(y_test_batch[i])
    
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_label_idx = np.argmax(pred)
    
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis("off")
    
    # Biru jika benar, Merah jika salah
    color = "blue" if pred_label_idx == true_label_idx else "red"
    plt.title(f"Prediksi: {class_names[pred_label_idx]}\nAsli: {class_names[true_label_idx]}", color=color)

plt.tight_layout()
plt.show()
print("\nSeluruh proses selesai!")