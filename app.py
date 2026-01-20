import os, json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import gdown

# ======================
# KONFIG
# ======================
IMG_SIZE = 128
MODEL_FILE_ID = "https://drive.google.com/file/d/1eScBrR1QndwXoL9-1X3xoO3LDawFxN7Q/view?usp=sharing"   # <-- WAJIB kamu ganti
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
MODEL_PATH = "model4_infer.keras"
CLASS_PATH = "class_names.json"

st.set_page_config(page_title="LeafVision", page_icon="🍃", layout="centered")

# ======================
# CUSTOM LAYER (karena model kamu pakai ini saat training)
# ======================
class RandomGamma(layers.Layer):
    def __init__(self, gamma_min=0.8, gamma_max=1.2, **kwargs):
        super().__init__(**kwargs)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def call(self, x, training=None):
        if training:
            gamma = tf.random.uniform([], self.gamma_min, self.gamma_max)
            x = tf.image.adjust_gamma(x, gamma=gamma)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma_min": self.gamma_min, "gamma_max": self.gamma_max})
        return cfg

class RandomGaussianNoise(layers.Layer):
    def __init__(self, stddev=0.03, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
        self.noise = layers.GaussianNoise(stddev)

    def call(self, x, training=None):
        return self.noise(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"stddev": self.stddev})
        return cfg

# ======================
# FUNCTIONS
# ======================
def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    with st.spinner("Model belum ada di server. Sedang download dari Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model_and_classes():
    download_model_if_needed()

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "RandomGamma": RandomGamma,
            "RandomGaussianNoise": RandomGaussianNoise
        },
        compile=False,
        safe_mode=False
    )
    return model, class_names

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def format_label(label: str):
    parts = label.split("___")
    plant = parts[0]
    disease = parts[1] if len(parts) > 1 else label
    status = "Sehat" if disease.lower() == "healthy" else "Penyakit"
    disease_pretty = "Healthy" if disease.lower() == "healthy" else disease.replace("_", " ")
    return plant, status, disease_pretty

# ======================
# UI (mirip app.php: upload -> hasil -> top3)
# ======================
st.markdown(
    """
    <style>
    .card {background: #1e293b; border: 1px solid rgba(255,255,255,0.10); padding: 18px; border-radius: 18px;}
    .muted {color: #cbd5e1;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍃 LeafVision | Deteksi Penyakit Daun")
st.write("Upload foto daun → sistem memprediksi **tanaman, status, diagnosis** + **Top-3**.")

model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Preview", use_container_width=True)

    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    label = class_names[pred_idx]
    plant, status, disease = format_label(label)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")
    st.write(f"**Tanaman:** {plant}")
    st.write(f"**Status:** {status}")
    st.write(f"**Diagnosis:** {disease}")
    st.write(f"**Confidence:** {conf:.4f}")
    st.progress(min(max(conf, 0.0), 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.subheader("Top-3 Prediksi")
    idxs = np.argsort(probs)[::-1][:3]
    for rank, i in enumerate(idxs, start=1):
        lab = class_names[int(i)]
        p, s, d = format_label(lab)
        st.write(f"{rank}. **{p}** | {s} | {d} — **{float(probs[int(i)]):.4f}**")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Silakan upload gambar daun untuk melihat hasil prediksi.")

