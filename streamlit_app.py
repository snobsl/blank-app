import streamlit as st

# Judul aplikasi
st.title("🎈 Sentiment Analysis Application")

# Deskripsi aplikasi
st.write(
    "Selamat datang di aplikasi sentimen analisis!"
)

# Input teks dari pengguna
user_input = st.text_area("Enter text for sentiment analysis:")
