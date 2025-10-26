# 🚀 Pipeline Analisis Sentimen Interaktif Bahasa Indonesia

Proyek ini menyediakan sebuah aplikasi web interaktif yang dibangun menggunakan Streamlit untuk melakukan analisis sentimen secara end-to-end pada data teks Bahasa Indonesia. Pengguna dapat mengunggah data CSV mereka sendiri, memprosesnya melalui tahapan pra-pemrosesan, memberi label sentimen secara otomatis menggunakan kamus lexicon, memvisualisasikan insight, melatih berbagai model machine learning dengan parameter khusus, serta menguji model yang telah dilatih menggunakan input baru.

---

##  latar belakang

Analisis sentimen, proses mengidentifikasi dan mengkategorikan opini yang diekspresikan dalam teks, menjadi semakin penting. Memahami sentimen pelanggan dari ulasan produk, komentar media sosial, atau survei dapat memberikan wawasan berharga untuk pengembangan produk, strategi pemasaran, dan manajemen reputasi. Namun, membangun *pipeline* analisis sentimen dari awal bisa jadi kompleks, melibatkan banyak langkah mulai dari pembersihan data hingga *deployment* model.

---

## Tujuan

Project ini bertujuan untuk:

1.  **Mendemonstrasikan** alur kerja *end-to-end* sebuah *pipeline* analisis sentimen sederhana.
2.  Menyediakan **alat interaktif** bagi pengguna untuk melakukan analisis sentimen pada data teks Bahasa Indonesia mereka sendiri.
3.  Memungkinkan **eksplorasi** pengaruh berbagai langkah preprocessing, metode ekstraksi fitur, dan model klasifikasi terhadap hasil akhir.
4.  Menjadi **portofolio** yang menunjukkan kemampuan dalam data preprocessing, NLP, *machine learning*, dan *web app development* menggunakan Streamlit.

---

## Fitur ✨

Aplikasi ini mencakup langkah-langkah berikut dalam *pipeline* analisis sentimen:

1.  **⬆️ Upload Data:** Pengguna dapat mengunggah file CSV yang berisi data teks.
2.  **🧹 Preprocessing Data:**
    * Pengguna memilih kolom teks yang akan dianalisis.
    * Teks dibersihkan secara otomatis (penghapusan emoji, mention, hashtag, URL, angka, tanda baca, spasi berlebih, *case folding*).
    * Normalisasi kata slang (menggunakan kamus sederhana).
    * Tokenisasi (memecah teks menjadi kata).
    * *Stopword removal* (menghapus kata umum Bahasa Indonesia & Inggris).
    * Penggabungan token kembali menjadi kalimat bersih (`text_final`).
    * Validasi data input (jumlah baris minimum, cek kolom kosong/tipe data).
3.  **🏷️ Pelabelan Otomatis (Lexicon):**
    * Pengguna memilih skema label (2 label: Positif/Negatif atau 3 label: Positif/Neutral/Negatif).
    * Sentimen ditentukan secara otomatis berdasarkan skor kata menggunakan lexicon Bahasa Indonesia (positif & negatif).
    * Opsi untuk mengunduh dataset lengkap yang telah melalui preprocessing dan pelabelan.
4.  **📊 Visualisasi & Insight:**
    * Menampilkan distribusi sentimen (Pie Plot) dari data yang telah dilabeli (dan difilter).
    * Menampilkan Word Cloud untuk kata-kata yang paling sering muncul di setiap kategori sentimen.
5.  **🤖 Modelling:**
    * Pengguna memilih metode ekstraksi fitur:
        * TF-IDF (Term Frequency-Inverse Document Frequency)
        * Bag-of-Words (BoW)
    * Pengguna memilih set parameter yang direkomendasikan untuk ekstraksi fitur (Default, Focused, Broad).
    * Pengguna memilih model klasifikasi:
        * Logistic Regression
        * SVM (Support Vector Machine)
        * Multinomial Naive Bayes
        * Bernoulli Naive Bayes
    * Pengguna memilih *hyperparameter* yang relevan untuk model yang dipilih (misalnya, `C` untuk LR/SVM, `alpha` untuk NB, `kernel` untuk SVM, `max_iter` untuk LR).
    * Model dilatih pada data latih (80%) dan dievaluasi pada data uji (20%).
    * Hasil evaluasi ditampilkan: Akurasi (Train & Test), Classification Report, dan Confusion Matrix.
    * Opsi untuk mengunduh artefak model (model, vectorizer, label encoder) dalam satu file ZIP.
6.  **🔍 Coba Prediksi (Inferensi):**
    * Pengguna dapat memasukkan teks/komentar baru.
    * Teks tersebut akan melalui *pipeline* preprocessing yang **sama persis** seperti saat training.
    * Model yang telah dilatih digunakan untuk memprediksi sentimen teks baru tersebut.
    * Hasil prediksi ditampilkan kepada pengguna.

---

## Keunggulan ✅

* **Interaktif & Visual:** Menggunakan Streamlit untuk antarmuka yang mudah digunakan dan visualisasi yang membantu pemahaman.
* **End-to-End:** Mencakup seluruh siklus dasar proyek *machine learning* dari data mentah hingga prediksi.
* **Kustomisasi:** Pengguna dapat mengontrol banyak aspek *pipeline* (pilihan fitur, model, parameter).
* **Fokus Bahasa Indonesia:** Dirancang khusus untuk teks Bahasa Indonesia, termasuk *stopwords* dan *lexicon* yang relevan.
* **Reproducibility:** Memungkinkan pengunduhan data yang diproses dan artefak model.
* **Edukasi:** Dapat digunakan sebagai alat belajar untuk memahami langkah-langkah dalam analisis sentimen.

---

## Keterbatasan ⚠️

* **Pelabelan Berbasis Lexicon:** Akurasi pelabelan sangat bergantung pada kualitas dan cakupan *lexicon*. Tidak dapat menangani sarkasme, konteks kalimat yang kompleks, atau kata-kata baru/domain spesifik yang tidak ada di *lexicon*.
* **Preprocessing Tetap:** Langkah-langkah preprocessing (seperti penghapusan tanda baca atau angka) bersifat tetap dan tidak dapat dikustomisasi oleh pengguna di UI saat ini. *Stemming* tidak diimplementasikan.
* **Kamus Slang Terbatas:** Kamus slang yang digunakan sangat sederhana dan perlu diperluas secara signifikan untuk kasus penggunaan nyata.
* **Parameter Terbatas:** Hanya beberapa *hyperparameter* utama yang diekspos ke pengguna. *Hyperparameter tuning* otomatis (seperti Grid Search) tidak diimplementasikan.
* **Skalabilitas:** Aplikasi ini dirancang untuk demonstrasi dan mungkin tidak efisien untuk dataset yang sangat besar (ratusan ribu atau jutaan baris) karena pemrosesan dilakukan di memori server Streamlit.
* **Bukan Produksi:** Tidak dimaksudkan untuk *deployment* skala produksi tanpa modifikasi lebih lanjut (misalnya, *error handling* yang lebih robust, *logging*, pemisahan *frontend* dan *backend*, dll.).

---

## Kontribusi

Saran dan kontribusi untuk perbaikan project ini sangat diterima. Silakan buat *issue* atau *pull request*.
