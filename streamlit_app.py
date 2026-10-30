from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🧬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_rf_model.joblib"

FEATURES = [
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
]
FEATURE_LABELS = {
    "Clump_thickness": "Clump thickness",
    "Uniformity_of_cell_size": "Uniformity of cell size",
    "Uniformity_of_cell_shape": "Uniformity of cell shape",
    "Marginal_adhesion": "Marginal adhesion",
    "Single_epithelial_cell_size": "Single epithelial cell size",
    "Bare_nuclei": "Bare nuclei",
    "Bland_chromatin": "Bland chromatin",
    "Normal_nucleoli": "Normal nucleoli",
    "Mitoses": "Mitoses",
}

# Hasil eksperimen dari notebook revisi pengguna.
RESULTS = {
    "Logistic Regression": {
        "Accuracy": 0.9500, "Precision": 0.9362, "Recall": 0.9167,
        "F1-Score": 0.9263, "ROC-AUC": 0.9950, "Average Score": 0.9448,
    },
    "Decision Tree": {
        "Accuracy": 0.9286, "Precision": 0.9318, "Recall": 0.8542,
        "F1-Score": 0.8913, "ROC-AUC": 0.9108, "Average Score": 0.9033,
    },
    "Random Forest": {
        "Accuracy": 0.9500, "Precision": 0.9184, "Recall": 0.9375,
        "F1-Score": 0.9278, "ROC-AUC": 0.9885, "Average Score": 0.9448,
    },
}
METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Average Score"]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, f"File model tidak ditemukan: {MODEL_PATH.name}"
    try:
        package = joblib.load(MODEL_PATH)
        if not isinstance(package, dict) or "model" not in package:
            return None, "Format file model tidak sesuai dengan paket model yang diharapkan."
        return package, None
    except Exception as exc:
        return None, f"Model gagal dimuat: {type(exc).__name__}: {exc}"

st.title("🧬 Breast Cancer Prediction System")
st.subheader("Perbandingan Algoritma Klasifikasi dan Prediksi")
st.write(
    "Dashboard pembelajaran mesin untuk dataset Breast Cancer Wisconsin (Original), "
    "dengan perbandingan Logistic Regression, Decision Tree, dan Random Forest."
)
st.info(
    "Hasil evaluasi di bawah berasal dari notebook revisi. Prediksi pada formulir "
    "menggunakan model Random Forest yang disimpan dalam file joblib."
)

st.header("1. Ringkasan hasil evaluasi")
metric_cols = st.columns(4)
summary = [
    ("Accuracy tertinggi", "95,00%", "Logistic Regression & Random Forest"),
    ("Recall tertinggi", "93,75%", "Random Forest"),
    ("F1-Score tertinggi", "92,78%", "Random Forest"),
    ("ROC-AUC tertinggi", "99,50%", "Logistic Regression"),
]
for col, (label, value, detail) in zip(metric_cols, summary):
    col.metric(label, value)
    col.caption(detail)

st.header("2. Perbandingan tiga algoritma")
metric_choice = st.selectbox("Pilih metrik evaluasi", METRICS, index=0)
chart_df = pd.DataFrame(
    {"Algoritma": list(RESULTS.keys()), metric_choice: [RESULTS[name][metric_choice] for name in RESULTS]}
).set_index("Algoritma")
st.bar_chart(df, y=metric_choice)

results_df = pd.DataFrame.from_dict(RESULTS, orient="index")
results_df.index.name = "Algoritma"
st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

st.subheader("Interpretasi singkat")
st.write(
    "Logistic Regression dan Random Forest memperoleh Average Score yang sama, yaitu 0,9448. "
    "Random Forest memiliki Recall tertinggi (0,9375), sedangkan Logistic Regression "
    "memiliki Precision (0,9362) dan ROC-AUC (0,9950) tertinggi. Pemilihan model "
    "bergantung pada metrik yang diprioritaskan dalam eksperimen."
)

with st.expander("Mengenal algoritma yang dibandingkan"):
    st.markdown(
        "- **Logistic Regression:** memperkirakan probabilitas kelas berdasarkan kombinasi fitur.\n"
        "- **Decision Tree:** membentuk aturan keputusan dalam struktur percabangan.\n"
        "- **Random Forest:** menggabungkan prediksi dari banyak pohon keputusan."
    )

st.header("3. Prediksi menggunakan Random Forest")
package, model_error = load_model()
if package is None:
    st.error(model_error or "Model belum tersedia.")
    st.warning(
        "Pastikan file `breast_cancer_rf_model.joblib` berada satu folder dengan "
        "`streamlit_app.py`, dan versi scikit-learn sesuai dengan versi saat model dilatih."
    )
else:
    expected_features = package.get("feature_names", FEATURES)
    if list(expected_features) != FEATURES:
        st.warning("Urutan/nama fitur dalam file model berbeda dari konfigurasi formulir. Periksa kembali model.")
    st.caption("Masukkan sembilan nilai fitur. Setiap nilai mengikuti rentang 1 sampai 10 pada dataset.")
    with st.form("prediction_form"):
        input_cols = st.columns(3)
        values = {}
        for idx, feature in enumerate(FEATURES):
            with input_cols[idx % 3]:
                values[feature] = st.number_input(
                    FEATURE_LABELS[feature], min_value=1, max_value=10,
                    value=5, step=1, key=f"input_{feature}"
                )
        submitted = st.form_submit_button("Jalankan prediksi", type="primary")

    if submitted:
        try:
            feature_order = list(package.get("feature_names", FEATURES))
            sample = pd.DataFrame([[values[name] for name in feature_order]], columns=feature_order)
            pipeline = package["model"]
            prediction = int(pipeline.predict(sample)[0])
            probabilities = pipeline.predict_proba(sample)[0]
            classifier = pipeline.named_steps["classifier"]
            classes = list(classifier.classes_)
            positive_index = classes.index(1)
            malignant_probability = float(probabilities[positive_index])
            predicted_label = "Malignant" if prediction == 1 else "Benign"

            st.subheader("Hasil klasifikasi model")
            if prediction == 1:
                st.warning(f"Kelas prediksi: **{predicted_label}**")
            else:
                st.success(f"Kelas prediksi: **{predicted_label}**")
            st.metric("Probabilitas kelas Malignant menurut model", f"{malignant_probability:.2%}")
            st.caption(
                "Hasil ini merupakan keluaran model klasifikasi untuk demonstrasi akademik. "
                "Bukan diagnosis medis dan tidak boleh digunakan untuk mengambil keputusan kesehatan."
            )
        except Exception as exc:
            st.error(f"Prediksi gagal dijalankan: {type(exc).__name__}: {exc}")

st.divider()
st.caption(
    "Dataset: Breast Cancer Wisconsin (Original) · Kegiatan pembelajaran Pembelajaran Mesin · "
    "Bukan alat diagnosis medis."
)
