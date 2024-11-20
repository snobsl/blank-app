#App With Streamlit
# Judul aplikasi
st.title("🎈 Sentiment Analysis Application")

# Deskripsi aplikasi
st.write(
    "Selamat datang di aplikasi sentimen analisis!"
)

# Input teks dari pengguna
user_input = st.text_area("Enter text for sentiment analysis:")

# Membersihkan teks
clean_input = clean_text(user_input)

# Mengubah teks menjadi fitur
input_tfidf = vectorizer.transform([clean_input])

# Prediksi dengan masing-masing model
nb_prediction = nb_model.predict(input_tfidf)
knn_prediction = knn_model.predict(input_tfidf)
svm_prediction = svm_model.predict(input_tfidf)

# Menampilkan hasil prediksi
st.write(f"Naive Bayes Prediction: {nb_prediction[0]}")
st.write(f"KNN Prediction: {knn_prediction[0]}")
st.write(f"SVM Prediction: {svm_prediction[0]}")
