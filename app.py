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
APP_NAME = "INDAH.PORT"
PERSON_NAME = "Indah Theresia"
LOCATION = "Pekanbaru, Indonesia"
MAJOR = "Teknik Informatika"

IMG_SIZE = 128

# === GANTI INI DENGAN FILE_ID DRIVE (bukan URL panjang) ===
MODEL_FILE_ID = "1eScBrR1QndwXoL9-1X3xoO3LDawFxN7Q"  # contoh: isi FILE_ID kamu
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
MODEL_PATH = "leaf_model.keras"
CLASS_PATH = "class_names.json"

st.set_page_config(page_title="LeafVision | Portfolio", page_icon="🍃", layout="wide")

# ======================
# THEME CSS (meniru style portofolio kamu)
# ======================
st.markdown("""
<style>
:root{
  --bg:#F7FAF8;
  --card:#FFFFFF;
  --card2:#F0F7F3;

  --accent:#22C55E;
  --accent2:#15803D;

  --text:#0F172A;
  --muted:#475569;

  --border: rgba(15,23,42,0.10);
  --shadow: 0 12px 30px rgba(2,6,23,0.08);
}

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp{ background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

a{ text-decoration:none; color: inherit; }

/* Navbar */
.navbarx{
  position: sticky; top: 0; z-index: 50;
  background: rgba(247,250,248,0.85);
  border: 1px solid var(--border);
  backdrop-filter: blur(10px);
  padding: 12px 18px;
  border-radius: 16px;
  box-shadow: 0 10px 18px rgba(2,6,23,0.06);
  margin-bottom: 18px;
}
.brand{ font-weight: 900; letter-spacing:.3px; font-size: 18px; color: var(--text); }
.brand span{ color: var(--accent2); }

/* Cards */
.card-soft{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: var(--shadow);
}
.hero-card{
  background: radial-gradient(circle at top right, rgba(34,197,94,0.18) 0%, var(--card) 60%);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 26px;
  box-shadow: var(--shadow);
}
.section-title{
  font-weight: 900;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--accent2);
  font-size: 14px;
  margin: 6px 0 14px;
}
.small-muted{ color: var(--muted); font-size: 0.95rem; }
.hero-title{ font-weight: 900; }
.accent{ color: var(--accent2); }

/* Chip */
.chip{
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(34,197,94,0.30);
  background: rgba(34,197,94,0.10);
  color: var(--text);
  font-weight: 700;
  margin-right: 8px; margin-top: 10px;
  font-size: 0.85rem;
}

/* Feature row */
.feature{
  display:flex; gap:12px;
  padding: 12px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: var(--card2);
  margin-bottom: 10px;
}
.feature .ico{
  width: 38px; height: 38px;
  display:flex; align-items:center; justify-content:center;
  border-radius: 12px;
  background: rgba(34,197,94,0.12);
  border: 1px solid rgba(34,197,94,0.25);
  color: var(--accent2);
  font-weight: 900;
}

/* Buttons (streamlit default button) */
div.stButton > button{
  border-radius: 999px !important;
  font-weight: 800 !important;
  border: 1px solid rgba(34,197,94,0.35) !important;
  background: rgba(34,197,94,0.14) !important;
  color: var(--text) !important;
}
div.stButton > button:hover{
  background: rgba(34,197,94,0.22) !important;
  border-color: rgba(21,128,61,0.55) !important;
  transform: translateY(-1px);
}

/* File uploader */
div[data-testid="stFileUploaderDropzone"]{
  background: var(--card2);
  border: 1px dashed rgba(34,197,94,0.40);
  border-radius: 18px;
}
div[data-testid="stFileUploaderDropzone"] *{
  color: var(--muted) !important;
}

/* Tables */
thead th{
  background: rgba(34,197,94,0.18) !important;
  color: var(--text) !important;
}
tbody td{
  background: rgba(255,255,255,0.90) !important;
  color: var(--text) !important;
}

/* Footer */
.footerx{
  margin-top: 26px;
  border-top: 1px solid var(--border);
  padding-top: 14px;
  color: var(--muted);
  text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ======================
# CUSTOM LAYER (karena model kamu pakai augmentasi custom)
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
# UTIL
# ======================
def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    with st.spinner("Model belum ada di server. Download dari Google Drive..."):
        # penting: MODEL_URL harus "uc?id=FILE_ID"
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

@st.cache_resource
def load_model_and_classes():
    download_model_if_needed()

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"RandomGamma": RandomGamma, "RandomGaussianNoise": RandomGaussianNoise},
        compile=False,
        safe_mode=False
    )
    return model, class_names

def preprocess_pil(img: Image.Image) -> np.ndarray:
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

def explain_label(label: str):
    """
    label contoh: 'Apple___Black_rot' atau 'Tomato___Leaf_Mold'
    return: (plant_pretty, status, simple_text)
    """
    parts = label.split("___", 1)
    plant = parts[0]
    disease_raw = parts[1] if len(parts) > 1 else label

    plant_pretty = plant.replace("_", " ").strip()
    disease_pretty = disease_raw.replace("_", " ").strip()

    # mapping sederhana (tidak melenceng: tetap menyebut istilah aslinya)
    disease_map = {
        "healthy": f"Daun {plant_pretty} terlihat sehat.",

        "Apple_scab": f"Daun {plant_pretty} terindikasi kudis/keropeng (apple scab).",
        "Black_rot": f"Daun {plant_pretty} terindikasi busuk hitam (black rot).",
        "Cedar_apple_rust": f"Daun {plant_pretty} terindikasi karat daun (cedar apple rust).",

        "Powdery_mildew": f"Daun {plant_pretty} terindikasi jamur embun tepung (powdery mildew).",

        "Cercospora_leaf_spot Gray_leaf_spot": f"Daun {plant_pretty} terindikasi bercak daun (Cercospora/Gray leaf spot).",
        "Common_rust": f"Daun {plant_pretty} terindikasi karat daun (common rust).",
        "Northern_Leaf_Blight": f"Daun {plant_pretty} terindikasi hawar daun (northern leaf blight).",

        "Esca_(Black_Measles)": f"Daun {plant_pretty} terindikasi penyakit Esca / black measles.",
        "Leaf_blight_(Isariopsis_Leaf_Spot)": f"Daun {plant_pretty} terindikasi bercak daun (Isariopsis leaf spot).",

        "Haunglongbing_(Citrus_greening)": f"Daun {plant_pretty} terindikasi HLB / citrus greening (huanglongbing).",

        "Bacterial_spot": f"Daun {plant_pretty} terindikasi bercak bakteri (bacterial spot).",
        "Early_blight": f"Daun {plant_pretty} terindikasi hawar awal (early blight).",
        "Late_blight": f"Daun {plant_pretty} terindikasi hawar akhir (late blight).",

        "Leaf_scorch": f"Daun {plant_pretty} terindikasi gejala 'daun terbakar' (leaf scorch).",

        "Leaf_Mold": f"Daun {plant_pretty} terindikasi jamur daun (leaf mold).",
        "Septoria_leaf_spot": f"Daun {plant_pretty} terindikasi bercak daun Septoria (septoria leaf spot).",
        "Spider_mites Two-spotted_spider_mite": f"Daun {plant_pretty} terindikasi serangan tungau (two-spotted spider mite).",
        "Target_Spot": f"Daun {plant_pretty} terindikasi bercak target (target spot).",
        "Tomato_Yellow_Leaf_Curl_Virus": f"Daun {plant_pretty} terindikasi virus kuning keriting (TYLCV).",
        "Tomato_mosaic_virus": f"Daun {plant_pretty} terindikasi virus mosaik (tomato mosaic virus).",
    }

    if disease_raw.lower() == "healthy":
        status = "Sehat"
        simple = disease_map["healthy"]
    else:
        status = "Terindikasi Penyakit"
        simple = disease_map.get(disease_raw, f"Daun {plant_pretty} terindikasi {disease_pretty}.")

    return plant_pretty, status, simple

# ======================
# NAV STATE
# ======================
if "page" not in st.session_state:
    st.session_state.page = "Portfolio"

def goto(page_name: str):
    st.session_state.page = page_name
    st.rerun()

# ======================
# NAVBAR (mirip portofolio)
# ======================
left, right = st.columns([1.2, 1])
with left:
    st.markdown(f"""
    <div class="navbarx">
      <div class="brand">🍃 {APP_NAME.split('.')[0]}<span>.{APP_NAME.split('.')[1]}</span></div>
    </div>
    """, unsafe_allow_html=True)

with right:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🏠 Portfolio", use_container_width=True):
            goto("Portfolio")
    with c2:
        if st.button("🍃 Deteksi Daun", use_container_width=True):
            goto("Deteksi Daun")

# ======================
# PAGE: PORTFOLIO
# ======================
if st.session_state.page == "Portfolio":
    colA, colB = st.columns([1.35, 1], gap="large")

    with colA:
        st.markdown(f"""
        <div class="hero-card">
          <div class="section-title">Home</div>
          <h1 class="hero-title">{PERSON_NAME}</h1>
          <div class="small-muted" style="font-size:1.05rem;">
            Fokus di <span class="accent" style="font-weight:800;">Web Dev</span>
          </div>
          <p class="small-muted" style="margin-top:12px;">
            Mahasiswa Teknik Informatika yang tertarik membangun solusi digital yang terukur:
            Web modern, Data Science, dan implementasi AI ke aplikasi nyata.
          </p>
        </div>
        """, unsafe_allow_html=True)

        btn1, btn2 = st.columns([1, 1])
        with btn1:
            if st.button("🧭 Lihat Project", use_container_width=True):
                st.info("Scroll ke bagian Projects di bawah (di Streamlit tidak 100% anchor).")
        with btn2:
            if st.button("🍃 Coba LeafVision", use_container_width=True):
                goto("Deteksi Daun")

        st.markdown("""
        <div style="margin-top:10px;">
          <span class="chip">🐍 Python</span>
          <span class="chip">🧠 TensorFlow</span>
          <span class="chip">🐘 PHP</span>
          <span class="chip">🧩 Bootstrap</span>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="card-soft">
          <div class="section-title">Highlight Project</div>
          <h3 style="margin-top:0;">✨ LeafVision</h3>

          <div class="feature">
            <div class="ico">🖼</div>
            <div>
              <div style="font-weight:800;">Upload Foto Daun</div>
              <div class="small-muted">Sistem menerima JPG/PNG dari user.</div>
            </div>
          </div>

          <div class="feature">
            <div class="ico">⚙</div>
            <div>
              <div style="font-weight:800;">Model CNN</div>
              <div class="small-muted">Prediksi PlantVillage 38 kelas (Top-3 + confidence).</div>
            </div>
          </div>

          <div class="feature">
            <div class="ico">🚀</div>
            <div>
              <div style="font-weight:800;">Deployable</div>
              <div class="small-muted">Model di-load langsung di web (tanpa PHP).</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Open LeafVision App", use_container_width=True):
            goto("Deteksi Daun")

    # ABOUT + QUICK INFO
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    a1, a2 = st.columns([1.3, 1], gap="large")
    with a1:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">Tentang Saya</h3>
          <p class="small-muted">
            Saya menyukai pembuatan sistem end-to-end: dari pengolahan data, training model,
            sampai deployment menjadi aplikasi yang bisa digunakan user.
          </p>
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-top:10px;">
            <div class="card-soft" style="background: rgba(15,23,42,0.6); padding:14px;">
              <div style="font-weight:800;">Web Dev</div><div class="small-muted">PHP/Laravel + UI</div>
            </div>
            <div class="card-soft" style="background: rgba(15,23,42,0.6); padding:14px;">
              <div style="font-weight:800;">AI / CV</div><div class="small-muted">CNN, TF/Keras</div>
            </div>
            <div class="card-soft" style="background: rgba(15,23,42,0.6); padding:14px;">
              <div style="font-weight:800;">Deployment</div><div class="small-muted">Web + Model</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class="card-soft">
          <h3 style="margin-top:0;">Quick Info</h3>
          <div class="small-muted">📍 {LOCATION}</div>
          <div class="small-muted">🎓 {MAJOR}</div>
          <div class="small-muted">⭐ Focus: Web, Data, AI</div>
        </div>
        """, unsafe_allow_html=True)

    # SKILLS
    st.markdown('<div class="section-title">Skills</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">💻 Development</h3>
          <span class="chip">PHP</span>
          <span class="chip">Laravel</span>
          <span class="chip">MySQL</span>
          <span class="chip">Bootstrap</span>
          <span class="chip">REST API</span>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">🧠 Data & AI</h3>
          <span class="chip">Python</span>
          <span class="chip">TensorFlow</span>
          <span class="chip">CNN</span>
          <span class="chip">Computer Vision</span>
          <span class="chip">Deployment</span>
        </div>
        """, unsafe_allow_html=True)

    # PROJECTS
    st.markdown('<div class="section-title">Projects</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="large")
    with p1:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">🍃 LeafVision</h3>
          <div class="small-muted">Klasifikasi penyakit daun (PlantVillage 38 kelas) + confidence + Top-3.</div>
          <div style="margin-top:8px;">
            <span class="chip">TFDS</span><span class="chip">CNN</span><span class="chip">Streamlit</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open LeafVision", use_container_width=True):
            goto("Deteksi Daun")
    with p2:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">⚙ Project Sistem</h3>
          <div class="small-muted">Isi deskripsi project lain kamu di sini.</div>
          <div style="margin-top:8px;"><span class="chip">Laravel</span><span class="chip">Database</span></div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">🔌 Project IoT</h3>
          <div class="small-muted">Isi deskripsi project IoT kamu di sini.</div>
          <div style="margin-top:8px;"><span class="chip">Arduino</span><span class="chip">Sensor</span></div>
        </div>
        """, unsafe_allow_html=True)

    # CONTACT
    st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">Let’s connect</h3>
          <div class="small-muted">Isi kontak asli kamu di sini.</div>
          <div class="small-muted" style="margin-top:10px;">✉ emailkamu@example.com</div>
          <div class="small-muted">💼 linkedin.com/in/username</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card-soft">
          <h3 style="margin-top:0;">Message (Demo)</h3>
        </div>
        """, unsafe_allow_html=True)
        st.text_input("Nama", placeholder="Nama kamu")
        st.text_input("Email", placeholder="email@kamu.com")
        st.text_area("Pesan", placeholder="Tulis pesan...")
        st.button("Kirim (Demo)", use_container_width=True)

    st.markdown(f'<div class="footerx">© {PERSON_NAME} | Built with ❤</div>', unsafe_allow_html=True)

def tomato_indices(class_names):
    return [i for i, name in enumerate(class_names) if name.startswith("Tomato___")]

def predict_tomato_only(model, class_names, img_pil, threshold=0.55):
    x = preprocess_pil(img_pil)
    probs = model.predict(x, verbose=0)[0]

    t_idx = tomato_indices(class_names)
    tomato_mass = float(np.sum(probs[t_idx]))  # total skor semua kelas Tomato

    best_local = int(np.argmax(probs[t_idx]))
    best_idx = int(t_idx[best_local])
    best_conf = float(probs[best_idx])

    is_tomato = tomato_mass >= threshold
    return is_tomato, tomato_mass, best_idx, best_conf

# ======================
# PAGE: DETEKSI DAUN
# ======================
else:
    st.markdown('<div class="section-title">LeafVision App</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Upload foto daun, sistem memprediksi tanaman + keterangan sederhana (confidence).</div>', unsafe_allow_html=True)

    # load model+class
    model, class_names = load_model_and_classes()

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Foto Daun")
        uploaded = st.file_uploader("Pilih JPG/PNG", type=["jpg", "jpeg", "png"])
        st.caption("Tips: daun terlihat dominan, tidak blur, pencahayaan cukup.")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded:
            img = Image.open(uploaded)
            st.markdown('<div class="card-soft" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown("### 🖼 Preview")
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown("### 📊 Hasil Prediksi")

        if not uploaded:
            st.info("Silakan upload gambar untuk melihat hasil prediksi.")
  else:
      img = Image.open(uploaded)
      is_tomato, tomato_mass, best_idx, best_conf = predict_tomato_only(
          model, class_names, img, threshold=0.55
      )
  
      if not is_tomato:
          st.error("Aplikasi ini hanya untuk **daun tomat**. Silakan upload foto daun tomat.")
          st.caption(f"(Skor tomat: {tomato_mass:.4f} — di bawah threshold)")
      else:
          label = class_names[best_idx]
          plant, status, simple_text = explain_label(label)
  
          st.write(f"**Tanaman:** {plant}")  # harusnya Tomato
          st.write(f"**Status:** {status}")
          st.write(f"**Keterangan:** {simple_text}")
  
          st.write(f"**Confidence:** {best_conf:.4f} ({best_conf*100:.1f}%)")
          st.progress(min(max(best_conf, 0.0), 1.0))
          st.caption(f"Skor tomat total: {tomato_mass:.4f}")
  

            st.caption("Catatan: Jika confidence rendah, kemungkinan gambar berbeda jauh dari data latih PlantVillage.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="footerx">© {PERSON_NAME} | LeafVision — CNN PlantVillage</div>', unsafe_allow_html=True)


