import streamlit as st
from google.cloud import vision
import io
import os
import tempfile
import google.generativeai as genai
import re

# 🔐 Login sederhana
def login():
    password = st.text_input("Masukkan password:", type="password")
    secret_password = os.environ.get("password")
    if password != secret_password:
        st.stop()

login()

# 🔑 Tulis kredensial Google Vision dari ENV ke file sementara
def write_credential_file():
    cred_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if cred_json:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as f:
            f.write(cred_json)
            f.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

write_credential_file()

@st.cache_resource
def get_vision_client():
    return vision.ImageAnnotatorClient()

# 🧠 Inisialisasi Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# 🔍 Fungsi OCR
def detect_handwritten_text(image_bytes):
    client = get_vision_client()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        st.error(f"Error dari Google Vision API: {response.error.message}")
        return ""
    return response.full_text_annotation.text

# 🧠 Prompt ke Gemini
def post_process_text(raw_text):
    prompt = f"""
Berikut adalah hasil OCR dari teks tulisan tangan:

"{raw_text}"
Kamu adalah AI yang bertugas merapikan hasil OCR tulisan tangan.
Tolong perbaiki struktur kalimat, ejaan, dan rapikan tata letak. Berikan nomor untuk tiap jawaban yang sesuai dengan ejaan berdasarkan sintaksis bahasa Indonesia yang baik dan benar.
"""
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    response = model.generate_content(prompt)
    return response.text.strip()

# 🧼 Bersihkan format Markdown
def remove_markdown_formatting(text):
    text = re.sub(r"\*\*\*(.*?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return text.strip()

# 🌟 UI Utama
st.title("📄 OCR Tulisan Tangan + AI Perapihan (Gemini)")

uploaded_file = st.file_uploader("Unggah gambar tulisan tangan (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, caption="🖼️ Gambar Unggahan", use_container_width=True)

    with st.spinner("🔍 Memproses OCR..."):
        text = detect_handwritten_text(bytes_data)

    if text:
        st.subheader("📃 Hasil OCR Mentah:")
        st.text_area("Teks hasil OCR", value=text, height=300)

        st.download_button(
            label="💾 Unduh hasil OCR sebagai TXT",
            data=text,
            file_name="hasil_ocr.txt",
            mime="text/plain"
        )

        if st.button("✨ Perbaiki Teks dengan AI (Gemini)"):
            with st.spinner("🧠 Memproses dengan Gemini..."):
                improved_text = post_process_text(text)
                cleaned_text = remove_markdown_formatting(improved_text)

            st.subheader("🎯 Pilih Format Teks:")
            format_option = st.radio("Tampilkan sebagai:", ["Teks Polos", "Markdown"])

            final_text = cleaned_text if format_option == "Teks Polos" else improved_text

            st.subheader("📄 Teks Setelah Diperbaiki:")
            st.text_area("Teks yang sudah dirapikan", value=final_text, height=400)

            st.download_button(
                label="💾 Unduh Teks Rapi",
                data=final_text,
                file_name="hasil_rapi.txt",
                mime="text/plain"
            )
    else:
        st.warning("⚠️ Tidak ada teks terdeteksi dari gambar.")
