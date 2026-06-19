# Proyek Klasifikasi Gambar: Rice Image Dataset

Submission ini menggunakan Rice Image Dataset dari Kaggle.

## Dataset

- Source: Kaggle - muratkokludataset/rice-image-dataset
- Dataset asli: 75.000 gambar
- Jumlah kelas: 5
- Kelas: Arborio, Basmati, Ipsala, Jasmine, Karacadag
- Dataset yang digunakan pada notebook: 12500 gambar
- Split: 70% train, 15% validation, 15% test

## Model

Model dibuat menggunakan TensorFlow/Keras dengan arsitektur Sequential CNN.
Model memenuhi kriteria penggunaan Sequential, Conv2D, dan MaxPooling2D.

## Hasil Evaluasi

- Training Accuracy: 0.9699
- Validation Accuracy: 0.9840
- Testing Accuracy: 0.9829

## Format Model

Folder submission berisi:

- saved_model/
- tflite/model.tflite
- tflite/label.txt
- tfjs_model/model.json
- tfjs_model/group1-shard*.bin
- README.md
- requirements.txt

## Inference

Notebook sudah menyertakan contoh inference menggunakan model Keras dan TF-Lite.
