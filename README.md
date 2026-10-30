# Breast Cancer Prediction System — Streamlit

Aplikasi Streamlit untuk menampilkan perbandingan Logistic Regression, Decision Tree, dan Random Forest serta menjalankan prediksi menggunakan model Random Forest tersimpan.

## Struktur file di GitHub

```text
breast-cancer-dashboard/
├── streamlit_app.py
├── breast_cancer_rf_model.joblib
├── requirements.txt
└── README.md
```

File `breast_cancer_rf_model.joblib` yang sudah ada di repository harus tetap dipertahankan dan berada di folder utama yang sama dengan `streamlit_app.py`.

## Menjalankan secara lokal

Gunakan Python 3.11, lalu jalankan:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy ke Streamlit Community Cloud

1. Push `streamlit_app.py`, `requirements.txt`, dan README ini ke repository GitHub.
2. Buka https://share.streamlit.io/ dan pilih **Create app**.
3. Pilih repository `Anjar-aps/breast-cancer-dashboard`.
4. Pilih branch `main`.
5. Isi **Main file path** dengan `streamlit_app.py`.
6. Pilih versi Python 3.11 jika opsi pemilihan versi tersedia, lalu deploy.

## Catatan model

Model dilatih dengan scikit-learn 1.6.1. Karena itu `requirements.txt` mengunci versi tersebut agar sesuai dengan lingkungan pelatihan. Jika versi Python/dependensi di hosting berubah, pengujian ulang diperlukan.

## Catatan penggunaan

Angka metrik merupakan hasil eksperimen notebook revisi. Prediksi adalah demonstrasi pembelajaran mesin, bukan diagnosis medis atau pengganti pemeriksaan tenaga kesehatan.
