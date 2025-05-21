import streamlit as st
from google.cloud import vision
import io
import os
import tempfile
import google.generativeai as genai

# Autentikasi login sederhana
def login():
    password = st.text_input("Masukkan password:", type="password")
    secret_password = os.environ.get("password")
    if password != secret_password:
        st.stop()

login()

# Tulis credential file sementara dari ENV
def write_credential_file():
    cred_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if cred_json:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as f:
            f.write(cred_json)
            f.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

write_credential_file()

# Inisialisasi client Vision API
@st.cache_resource
def get_vision_client():
    return vision.ImageAnnotatorClient()

# Tampilkan path credential (debug opsional)
st.write("Path ke credential file:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

# Fungsi deteksi teks tulisan tangan dengan Vision API
def detect_handwritten_text(image_bytes):
    client = get_vision_client()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        st.error(f"Error dari Google Vision API: {response.error.message}")
        return ""
    return response.full_text_annotation.text

# Konfigurasi Gemini AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def post_process_text(raw_text):
    prompt = f"""
Kamu adalah AI yang bertugas merapikan hasil OCR tulisan tangan.

Tolong rapikan, perjelas, dan susun ulang teks berikut dalam format yang mudah dibaca,
dengan bahasa baku dan struktur yang baik. Jangan gunakan format markdown.

Teks:
\"\"\"
{raw_text}
\"\"\"
"""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# Antarmuka Streamlit
st.title("📷 OCR Tulisan Tangan + AI Perapihan (Google Cloud + Gemini)")

uploaded_file = st.file_uploader("Unggah gambar tulisan tangan (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, caption="📸 Gambar Unggahan", use_container_width=True)

    with st.spinner("🔍 Memproses OCR..."):
        text = detect_handwritten_text(bytes_data)

    if text:
        st.subheader("📄 Hasil OCR Mentah:")
        st.text_area("Teks hasil OCR", value=text, height=300)

        st.download_button(
            label="💾 Unduh hasil OCR (txt)",
            data=text,
            file_name="hasil_ocr.txt",
            mime="text/plain"
        )

        if st.button("✨ Perbaiki Teks dengan AI (Gemini)"):
            with st.spinner("⚙️ Memproses dengan Gemini..."):
                improved_text = post_process_text(text)

            st.subheader("🧠 Teks Setelah Diperbaiki:")
            st.text_area("Teks hasil AI", value=improved_text, height=400)

            st.download_button(
                label="💾 Unduh hasil perbaikan",
                data=improved_text,
                file_name="hasil_rapi.txt",
                mime="text/plain"
            )
    else:
        st.warning("⚠️ Tidak ada teks terdeteksi.")
