# Breast Cancer Prediction System

## Menjalankan lokal
Gunakan Python 3.11. Jalankan `pip install -r requirements.txt`, lalu `python app.py`. Buka `http://127.0.0.1:5000`.

## Deploy online
Unggah isi folder ini ke repository GitHub. Di Render pilih **New > Blueprint**, hubungkan repository, dan Render akan membaca `render.yaml`. Setelah deployment selesai, buka URL publik yang diberikan Render.

GitHub Pages tidak dapat menjalankan backend Flask. Aplikasi prediksi membutuhkan hosting Python.

## Endpoint
- `GET /health`: status aplikasi dan model.
- `POST /api/predict`: prediksi sembilan fitur dan probabilitas kelas Malignant.

## Catatan
File joblib harus berasal dari sumber tepercaya. Model ini untuk demonstrasi akademik, bukan diagnosis medis.
