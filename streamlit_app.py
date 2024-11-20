#1.Baca Data
import pandas as pd

# Membaca dataset
data = pd.read_csv('labeled_text (1).csv')

# Menampilkan beberapa baris pertama dari dataset
data.head()

import streamlit as st

#-----

#2.Membersihkan Data dan Membagi Data
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Membersihkan teks dalam dataset
data['clean_text'] = data['translated_text'].apply(clean_text)

# Memisahkan fitur dan label
X = data['clean_text']
y = data['sentimen']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengubah teks menjadi fitur menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#-----

#3.Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Melatih model Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Memprediksi pada set pengujian
y_pred_nb = nb_model.predict(X_test_tfidf)

# Akurasi Model
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Evaluasi model
print("Naive Bayes Classification Report")
print("Akurasi Naive Bayes:", accuracy_nb)
print(classification_report(y_test, y_pred_nb))

#4.KNN
from sklearn.neighbors import KNeighborsClassifier

# Melatih model KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_tfidf, y_train)

# Memprediksi pada set pengujian
y_pred_knn = knn_model.predict(X_test_tfidf)

# Akurasi Model
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Evaluasi model
print("KNN Classification Report")
print("Akurasi KNN:", accuracy_knn)
print(classification_report(y_test, y_pred_knn))

#-----

#5.SVM
from sklearn.svm import SVC

# Melatih model SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Memprediksi pada set pengujian
y_pred_svm = svm_model.predict(X_test_tfidf)

# Akurasi Model
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Evaluasi model
print("SVM Classification Report")
print("Akurasi SVM:", accuracy_svm)
print(classification_report(y_test, y_pred_svm))

#-----

#App With Streamlit
# Judul aplikasi
st.title("📊 Aplikasi Sentimen Analisis")

# Deskripsi aplikasi
st.write(
    "Selamat datang di aplikasi sentimen analisis!"
)
st.write(
    "Karya Leonard Agust Laga Lajar"
)
st.write(
    "Untuk melakukan sentimen analisis, silahkan masukkan teks."
)

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks untuk analisis sentimen:")

# Membersihkan teks
clean_input = clean_text(user_input)

# Mengubah teks menjadi fitur
input_tfidf = vectorizer.transform([clean_input])

# Prediksi dengan masing-masing model
nb_prediction = nb_model.predict(input_tfidf)
knn_prediction = knn_model.predict(input_tfidf)
svm_prediction = svm_model.predict(input_tfidf)

# Menampilkan hasil prediksi
# Naive Bayes
st.subheader("1. Naive Bayes")
st.write(f"Akurasi Naive Bayes: {accuracy_nb}")
st.write(f"Prediksi Naive Bayes: {nb_prediction[0]}")

if nb_prediction == "positif":
    st.success("Kesimpulan dari Prediksi dengan Naive Bayes: Gojek masih dapat berkembang 📈")
elif nb_prediction == "netral":
    st.info("Kesimpulan dari Prediksi dengan Naive Bayes: Gojek masih ada di posisi aman, tetapi dapat menurun sewaktu-waktu📊")
else:
    st.warning("Kesimpulan dari Prediksi dengan Naive Bayes: Gojek dapat menurun sewaktu-waktu📉")

# KNN
st.subheader("2. KNN")
st.write(f"Akurasi KNN: {accuracy_knn}")
st.write(f"KNN Prediction: {knn_prediction[0]}")

if knn_prediction == "positif":
    st.success("Kesimpulan dari Prediksi dengan KNN: Gojek masih dapat berkembang 📈")
elif knn_prediction == "netral":
    st.info("Kesimpulan dari Prediksi dengan KNN: Gojek masih ada di posisi aman, tetapi dapat menurun sewaktu-waktu📊")
else:
    st.warning("Kesimpulan dari Prediksi dengan KNN: Gojek dapat menurun sewaktu-waktu📉")


# SVM
st.subheader("3. SVM")
st.write(f"Akurasi SVM: {accuracy_svm}")
st.write(f"SVM Prediction: {svm_prediction[0]}")

if svm_prediction == "positif":
    st.success("Kesimpulan dari Prediksi dengan SVM: Gojek masih dapat berkembang 📈")
elif svm_prediction == "netral":
    st.info("Kesimpulan dari Prediksi dengan SVM: Gojek masih ada di posisi aman, tetapi dapat menurun sewaktu-waktu📊")
else:
    st.warning("Kesimpulan dari Prediksi dengan SVM: Gojek dapat menurun sewaktu-waktu📉")
