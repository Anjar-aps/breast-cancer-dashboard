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

# Identitas visual: feminine editorial, tetap nyaman dibaca untuk presentasi akademik.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root { --berry:#754B65; --rose:#B87996; --blush:#F6E5EC; --lilac:#EDE7F3; --sage:#719783; --ink:#493B45; }
.stApp { background: radial-gradient(circle at 4% 7%, rgba(231,190,208,.22), transparent 24%), #FBF5F7; color:var(--ink); }
.block-container { max-width:1180px; padding-top:2rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:'Playfair Display',Georgia,serif !important; color:#57394B !important; letter-spacing:-.02em; }
p,label { font-family:'DM Sans','Segoe UI',sans-serif; }
.hero-panel { background:linear-gradient(118deg,#70465F 0%,#95677F 58%,#B9879B 100%); color:white; padding:2.5rem 2.7rem; border-radius:24px; border-bottom:5px solid #D9A7B9; position:relative; overflow:hidden; margin-bottom:1.3rem; }
.hero-panel:after { content:'✿'; position:absolute; right:5%; top:-55px; font-size:210px; color:rgba(255,255,255,.10); }
.hero-kicker { font-size:.72rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700; opacity:.88; }
.hero-title { font-family:'Playfair Display',Georgia,serif; font-size:2.45rem; line-height:1.12; margin:.8rem 0 .7rem; color:white; max-width:780px; }
.hero-copy { color:rgba(255,255,255,.86); line-height:1.7; max-width:760px; font-size:.96rem; }
.section-kicker { color:#A16D87; text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:700; margin:1.3rem 0 .25rem; }
div[data-testid="stMetric"] { background:#FFFCFD; border:1px solid #EBD8E2; border-radius:18px; padding:1rem 1.1rem; box-shadow:0 7px 22px rgba(116,73,99,.055); min-height:118px; }
div[data-testid="stMetricLabel"] { color:#897783; font-weight:600; }
div[data-testid="stMetricValue"] { color:#754B65; font-family:'Playfair Display',Georgia,serif; }
div[data-testid="stAlert"] { border-radius:14px; }
div[data-testid="stForm"] { background:#FFFCFD; border:1px solid #EBD8E2; padding:1.2rem; border-radius:18px; }
.stButton>button,.stFormSubmitButton>button { background:linear-gradient(115deg,#754B65,#9B6B86); color:white; border:0; border-radius:999px; padding:.55rem 1.25rem; font-weight:700; }
.stButton>button:hover,.stFormSubmitButton>button:hover { background:#57394B; color:white; border:0; }
div[data-testid="stDataFrame"] { border:1px solid #EBD8E2; border-radius:14px; overflow:hidden; }
div[data-testid="stSelectbox"]>div>div { border-radius:12px; }
.soft-card { background:linear-gradient(125deg,#F8E9EF,#F0EAF5); border:1px solid #EBD5E0; border-left:5px solid #B87996; padding:1.15rem 1.3rem; border-radius:15px; margin:.5rem 0 1rem; color:#57394B; line-height:1.7; }
.algo-card { background:#FFFCFD; border:1px solid #EBD8E2; border-radius:16px; padding:1.1rem 1.2rem; min-height:190px; }
.algo-card h4 { font-family:'Playfair Display',Georgia,serif; color:#754B65; margin:.1rem 0 .55rem; font-size:1.15rem; }
.algo-card p { color:#756875; font-size:.88rem; line-height:1.65; }
.small-label { color:#A16D87; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.14em; }
hr { border-color:#EBD8E2; }
@media(max-width:700px){.hero-panel{padding:1.7rem 1.35rem}.hero-title{font-size:1.8rem}.hero-panel:after{font-size:130px;right:-3%}}
</style>
<div class="hero-panel">
 <div class="hero-kicker">Pembelajaran Mesin · Research Dashboard</div>
 <div class="hero-title">Breast Cancer Classification Study</div>
 <div class="hero-copy">Membandingkan Logistic Regression, Decision Tree, dan Random Forest pada dataset Breast Cancer Wisconsin (Original), lalu menguji prediksi menggunakan model Random Forest.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="soft-card"><b>Catatan eksperimen</b><br>Angka evaluasi ditampilkan sesuai hasil notebook revisi. Formulir prediksi menggunakan artefak Random Forest dari file joblib. Dashboard ini merupakan media pembelajaran, bukan alat diagnosis medis.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-kicker">01 / Model performance</div>', unsafe_allow_html=True)
st.header("Ringkasan hasil evaluasi")
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

st.markdown('<div class="section-kicker">02 / Comparative analysis</div>', unsafe_allow_html=True)
st.header("Perbandingan tiga algoritma")
metric_choice = st.selectbox("Pilih metrik evaluasi", METRICS, index=0)
chart_df = pd.DataFrame(
    {"Algoritma": list(RESULTS.keys()), metric_choice: [RESULTS[name][metric_choice] for name in RESULTS]}
).set_index("Algoritma")
st.bar_chart(chart_df, y=metric_choice, color="#B87996")

results_df = pd.DataFrame.from_dict(RESULTS, orient="index")
results_df.index.name = "Algoritma"
st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

st.markdown('<div class="section-kicker">03 / Reading the results</div>', unsafe_allow_html=True)
st.header("Interpretasi setiap algoritma")
algo_cols = st.columns(3)
with algo_cols[0]:
    st.markdown('<div class="algo-card"><div class="small-label">Model 01</div><h4>Logistic Regression</h4><p>Accuracy 0,9500, Precision 0,9362, dan ROC-AUC 0,9950. Model ini memiliki ROC-AUC tertinggi pada eksperimen, yang menunjukkan kemampuan pemisahan kelas yang sangat baik.</p></div>', unsafe_allow_html=True)
with algo_cols[1]:
    st.markdown('<div class="algo-card"><div class="small-label">Model 02</div><h4>Decision Tree</h4><p>Accuracy 0,9286 dan Recall 0,8542. Aturan pohon relatif mudah ditelusuri, tetapi Recall-nya paling rendah di antara ketiga model pada hasil eksperimen ini.</p></div>', unsafe_allow_html=True)
with algo_cols[2]:
    st.markdown('<div class="algo-card"><div class="small-label">Model 03</div><h4>Random Forest</h4><p>Recall 0,9375 dan F1-Score 0,9278, keduanya tertinggi. Model ini menjadi kandidat kuat ketika prioritas evaluasi adalah menangkap sebanyak mungkin kelas Malignant.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="soft-card"><b>Kesimpulan komparatif</b><br>Logistic Regression dan Random Forest sama-sama memperoleh Average Score 0,9448. Logistic Regression unggul pada Precision dan ROC-AUC, sedangkan Random Forest unggul pada Recall dan F1-Score. Tidak ada satu algoritma yang unggul pada seluruh metrik; pemilihan model perlu mengikuti tujuan penggunaan.</div>', unsafe_allow_html=True)

with st.expander("Mengenal algoritma yang dibandingkan"):
    st.markdown(
        "- **Logistic Regression:** memperkirakan probabilitas kelas berdasarkan kombinasi fitur; fitur distandardisasi pada eksperimen ini.\\n"
        "- **Decision Tree:** membentuk aturan keputusan dalam struktur percabangan.\\n"
        "- **Random Forest:** menggabungkan prediksi dari banyak pohon keputusan."
    )

st.markdown('<div class="section-kicker">04 / Experimental protocol</div>', unsafe_allow_html=True)
st.header("Metodologi eksperimen")
st.markdown("""
<div class="algo-card">
<ol style="line-height:1.9;color:#655563;margin:.2rem 0 .2rem 1rem">
<li><b>Dataset:</b> Breast Cancer Wisconsin (Original), 699 observasi dan 9 fitur karakteristik sel.</li>
<li><b>Target:</b> kelas Benign dan Malignant; label asli 2 dan 4 dipetakan menjadi 0 dan 1.</li>
<li><b>Preprocessing:</b> nilai hilang pada <i>Bare_nuclei</i> ditangani menggunakan median imputation. Imputer dipasang pada data training untuk mencegah kebocoran informasi.</li>
<li><b>Pemodelan:</b> membandingkan Logistic Regression, Decision Tree, dan Random Forest.</li>
<li><b>Evaluasi:</b> menggunakan Accuracy, Precision, Recall, F1-Score, dan ROC-AUC pada data pengujian.</li>
<li><b>Prediksi interaktif:</b> model Random Forest disimpan sebagai artefak joblib dan digunakan untuk memproses sembilan input fitur.</li>
</ol></div>
""", unsafe_allow_html=True)
st.caption("Rasio pembagian data dan parameter model mengikuti notebook eksperimen. Cantumkan rincian tersebut dari notebook saat menyusun laporan.")

st.markdown('<div class="section-kicker">05 / Interactive prediction</div>', unsafe_allow_html=True)

st.header("Prediksi menggunakan Random Forest")
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
    st.caption("Masukkan sembilan karakteristik sel. Setiap fitur menggunakan rentang nilai 1 sampai 10 sesuai dataset.")
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
