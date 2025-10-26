import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
from io import StringIO, BytesIO
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import zipfile

# Impor semua library Scikit-learn untuk Modelling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =======================================================================
# KONFIGURASI HALAMAN & SUMBER DAYA (CACHE)
# =======================================================================

# Konfigurasi halaman
st.set_page_config(page_title="Sentiment Pipeline", layout="wide")
st.title("Pipeline Analisis Sentimen Interaktif")
st.markdown("---")

# Set style untuk plot
sns.set(style="whitegrid")

@st.cache_resource
def setup_nltk_data():
    """Mencoba mengunduh data NLTK jika belum ada."""
    nltk_data_packages = ['punkt', 'stopwords', 'punkt_tab']
    all_successful = True
    for package in nltk_data_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package.startswith('punkt') else f'corpora/{package}')
            print(f"NLTK package '{package}' already downloaded.")
        except LookupError:
            print(f"NLTK package '{package}' not found. Attempting download...")
            try:
                nltk.download(package, quiet=True)
                print(f"Successfully downloaded NLTK package '{package}'.")
                nltk.data.find(f'tokenizers/{package}' if package.startswith('punkt') else f'corpora/{package}')
            except Exception as e:
                st.error(f"Gagal mengunduh NLTK package '{package}': {e}. Fitur terkait mungkin tidak berfungsi.")
                all_successful = False 
    return all_successful

nltk_setup_successful = setup_nltk_data()

@st.cache_resource
def get_stopwords_list():
    """Memuat dan meng-cache daftar stopwords."""
    try:
        listStopwords = set(stopwords.words('indonesian'))
        listStopwords.update(set(stopwords.words('english')))
        listStopwords.update(['iya','yaa','yaaa','gak','nya','na','sih','ku',"di","ga","ya",
                              "gaa","loh","kah","woi","woii","woy","ny","ko","klo","kalo"])
        return listStopwords
    except LookupError:
        st.error("Gagal memuat NLTK stopwords. Pastikan paket 'stopwords' terunduh.")
        return None
    except Exception as e:
        st.error(f"Error tidak terduga saat memuat stopwords: {e}")
        return None


LIST_STOPWORDS = get_stopwords_list()
SLANGWORDS = {"@": "di", "abis": "habis"}

@st.cache_resource
def load_lexicon_positive():
    lexicon_positive = dict()
    url = 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        reader = csv.reader(StringIO(response.text), delimiter=',')
        count = 0
        for row in reader:
            if len(row) >= 2:
                try:
                    lexicon_positive[row[0]] = int(row[1])
                    count += 1
                except ValueError:
                    continue 
        print(f"Loaded {count} positive words")
        if count == 0:
             st.warning("Tidak ada kata positif yang berhasil dimuat dari lexicon.")
        return lexicon_positive
    except requests.exceptions.RequestException as e:
        st.error(f"Error jaringan saat mengambil lexicon positif: {e}")
        return None
    except Exception as e:
        st.error(f"Error saat memproses lexicon positif: {e}")
        return None

@st.cache_resource
def load_lexicon_negative():
    lexicon_negative = dict()
    url = 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv'
    try:
        response = requests.get(url, timeout=10) 
        response.raise_for_status() 
        reader = csv.reader(StringIO(response.text), delimiter=',')
        count = 0
        for row in reader:
            if len(row) >= 2:
                try:
                    lexicon_negative[row[0]] = int(row[1])
                    count += 1
                except ValueError:
                    continue 
        print(f"Loaded {count} negative words")
        if count == 0:
            st.warning("Tidak ada kata negatif yang berhasil dimuat dari lexicon.")
        return lexicon_negative
    except requests.exceptions.RequestException as e:
        st.error(f"Error jaringan saat mengambil lexicon negatif: {e}")
        return None
    except Exception as e:
        st.error(f"Error saat memproses lexicon negatif: {e}")
        return None


# =======================================================================
# FUNGSI-FUNGSI PREPROCESSING, PELABELAN & MODELLING
# =======================================================================

def cleaningText(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U00002500-\U00002BEF"  # box drawing, geometric shapes
        "\U00010000-\U0010ffff"  # supplementary plane
        "]+",
        flags=re.UNICODE,
    )
    text = EMOJI_PATTERN.sub(r'', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def fix_slangwords(text, slang_dict):
    if not isinstance(slang_dict, dict): return text 
    words = text.split()
    fixed_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(fixed_words)

def tokenizingText(text):
    try:
        # jalankan tokenisasi
        tokens = word_tokenize(text)
        return tokens
    except LookupError:
        st.error("Data NLTK 'punkt' diperlukan untuk tokenisasi tidak ditemukan. Coba refresh halaman.")
        try:
             nltk.download('punkt', quiet=True)
             return word_tokenize(text)
        except Exception as download_error:
             st.error(f"Gagal mengunduh 'punkt' lagi: {download_error}")
             raise RuntimeError("NLTK 'punkt' download failed, cannot tokenize.")
    except Exception as e:
        st.error(f"Error saat tokenisasi: {e}")
        raise e 

def filteringText(text_tokens, stop_list):
    if stop_list is None:
        return text_tokens 
    if not isinstance(text_tokens, list):
         return [] 
    return [word for word in text_tokens if word not in stop_list]

def toSentence(list_words):
    if not isinstance(list_words, list):
        return ""
    return ' '.join(list_words)
    
@st.cache_data
def run_full_preprocessing(df, column_name):
    """
    Menjalankan pipeline preprocessing lengkap pada DataFrame.
    Mengembalikan DataFrame ASLI dengan SEMUA kolom baru hasil proses.
    """
    processed_df = df.copy()
    stop_list = get_stopwords_list()
    
    slang_dict = SLANGWORDS
    
    try:
        # Cleaning
        processed_df['text_clean'] = processed_df[column_name].astype(str).apply(cleaningText)
        # Fix slangwords
        processed_df['text_slang'] = processed_df['text_clean'].apply(lambda x: fix_slangwords(x, slang_dict))
        # Tokenizing
        processed_df['text_token'] = processed_df['text_slang'].apply(tokenizingText)
        # Stopword removal
        processed_df['text_stopword'] = processed_df['text_token'].apply(lambda x: filteringText(x, stop_list))
        # Convert to sentence
        processed_df['text_final'] = processed_df['text_stopword'].apply(toSentence)
        # Hitung panjang kata
        processed_df['text_length'] = processed_df['text_final'].apply(lambda x: len(x.split()))
        
        # Kembalikan DataFrame
        return processed_df
    except LookupError as e:
        st.error(f"Error NLTK saat preprocessing: {e}. Pastikan data NLTK terunduh. Coba refresh halaman.")
        return None
    except Exception as e: 
        st.error(f"Error tidak terduga saat preprocessing: {e}")
        st.exception(e) 
        return None


def sentiment_analysis_lexicon_indonesia(text, num_labels, lexicon_pos, lexicon_neg):
    if pd.isna(text) or text.strip() == '':
        return 0, 'neutral' if num_labels == 3 else 'negative'
    if isinstance(text, str):
        words = text.split()
    elif isinstance(text, list):
        words = text
    else:
         return 0, 'neutral' if num_labels == 3 else 'negative'

    score = 0
    if lexicon_pos:
        for word in words:
            score += lexicon_pos.get(word, 0) 
    if lexicon_neg:
        for word in words:
            score += lexicon_neg.get(word, 0) 
                
    if num_labels == 2:
        sentiment = 'positive' if score > 0 else 'negative'
    else:
        if score > 0: sentiment = 'positive'
        elif score < 0: sentiment = 'negative'
        else: sentiment = 'neutral'
    return score, sentiment

@st.cache_data
def run_labeling(df, text_column, num_labels, lex_pos, lex_neg):
    """
    Menerapkan pelabelan sentimen ke DataFrame.
    Streamlit akan mencoba hash konten df, lex_pos, lex_neg.
    """
    df_labeled = df.copy()
    
    if lex_pos is None or lex_neg is None:
        st.error("Lexicon tidak valid (gagal dimuat), pelabelan dibatalkan.")
        return None 
        
    try:
        results = df_labeled[text_column].apply(
            lambda text: sentiment_analysis_lexicon_indonesia(text, num_labels, lex_pos, lex_neg)
        )
        results_unpacked = list(zip(*results))
        if len(results_unpacked) == 2: 
             df_labeled['polarity_score'] = results_unpacked[0]
             df_labeled['sentiment'] = results_unpacked[1]
             return df_labeled
        else:
             st.error("Hasil pelabelan tidak sesuai format yang diharapkan.")
             return None
    except Exception as e:
        st.error(f"Error saat menerapkan pelabelan: {e}")
        st.exception(e)
        return None


@st.cache_data
def train_model(_df, text_column, label_column, feature_method, feature_params, model_name, model_params):
    """
    Melatih model machine learning berdasarkan pilihan user.
    Mengembalikan artefak model dan metrik evaluasi.
    """
    if _df is None or _df.empty:
        st.error("Data input untuk training model tidak valid atau kosong.")
        return None, None, None, 0, 0, "Error: Data training tidak valid", np.array([]) 

    # Persiapan Data
    X = _df[text_column]
    y = _df[label_column]

    # Cek jika label hanya punya 1 nilai unik
    unique_labels = y.nunique()
    stratify_param = y if unique_labels > 1 else None
    
    # Label Encoding
    le = LabelEncoder()
    try:
        y_encoded = le.fit_transform(y)
    except Exception as e:
        st.error(f"Error saat Label Encoding: {e}")
        return None, None, None, 0, 0, f"Error Label Encoding: {e}", np.array([])
    
    # Split Data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param 
        )
    except ValueError as e:
         # error jika stratify gagal (data per kelas < 2)
         st.warning(f"Gagal melakukan stratified split (mungkin karena data per kelas terlalu sedikit): {e}. Mencoba split biasa.")
         try:
            X_train, X_test, y_train, y_test = train_test_split(
                 X, y_encoded, test_size=0.2, random_state=42
            )
         except Exception as split_e:
             st.error(f"Error saat split data: {split_e}")
             return None, None, None, 0, 0, f"Error Split Data: {split_e}", np.array([])

    if X_train.empty or X_test.empty:
         st.error("Data train atau test kosong setelah split. Mungkin data awal terlalu sedikit.")
         return None, None, None, 0, 0, "Error: Data train/test kosong", np.array([])
    
    # Ekstraksi Fitur
    try:
        if feature_method == "TF-IDF":
            vectorizer = TfidfVectorizer(
                max_features=feature_params['max_features'],
                min_df=feature_params['min_df'],
                max_df=feature_params['max_df']
            )
        else: # Bag-of-Words (BoW)
            vectorizer = CountVectorizer(
                max_features=feature_params['max_features'],
                min_df=feature_params['min_df'],
                max_df=feature_params['max_df']
            )
        
        X_train_vec = vectorizer.fit_transform(X_train.astype('U')) 
        X_test_vec = vectorizer.transform(X_test.astype('U'))     
    except Exception as e:
        st.error(f"Error saat Ekstraksi Fitur ({feature_method}): {e}")
        return None, None, le, 0, 0, f"Error Ekstraksi Fitur: {e}", np.array([])
    
    # Training Model
    try:
        if model_name == "Logistic Regression":
            model = LogisticRegression(
                C=model_params['C_lr'], 
                max_iter=model_params['max_iter_lr'], 
                random_state=42
            )
        elif model_name == "SVM (Support Vector Machine)":
            model = SVC(
                C=model_params['C_svm'], 
                kernel=model_params['kernel_svm'], 
                random_state=42,
                probability=True 
            )
        elif model_name == "Multinomial Naive Bayes":
             if (X_train_vec < 0).sum() > 0 or (X_test_vec < 0).sum() > 0:
                  st.error("MultinomialNB tidak bisa menangani input negatif (mungkin dari TF-IDF?). Coba BoW atau model lain.")
                  return None, vectorizer, le, 0, 0, "Error: Input negatif untuk MultinomialNB", np.array([])
             model = MultinomialNB(alpha=model_params['alpha_mnb'])
        
        elif model_name == "Bernoulli Naive Bayes":
             model = BernoulliNB(alpha=model_params['alpha_bnb'])
            
        model.fit(X_train_vec, y_train)
    except Exception as e:
        st.error(f"Error saat training model {model_name}: {e}")
        return None, vectorizer, le, 0, 0, f"Error Training Model: {e}", np.array([])
    
    # 6. Evaluasi Model
    try:
        # Evaluasi Data Train
        y_pred_train = model.predict(X_train_vec)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        
        # Evaluasi Data Test
        y_pred_test = model.predict(X_test_vec)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        
        # Nama kelas dari LabelEncoder
        class_names = le.classes_ if hasattr(le, 'classes_') else None
        
        try:
            report = classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0)
        except ValueError as report_error:
            st.warning(f"Gagal membuat classification report lengkap: {report_error}. Mungkin hanya 1 kelas terdeteksi/diprediksi.")
            report = f"Accuracy: {accuracy_test:.4f}\n(Classification report tidak tersedia)"

        cm = confusion_matrix(y_test, y_pred_test, labels=le.transform(le.classes_) if class_names is not None else None) 
        
        return model, vectorizer, le, accuracy_train, accuracy_test, report, cm
    except Exception as e:
         st.error(f"Error saat evaluasi model: {e}")
         return model, vectorizer, le, accuracy_train, 0, f"Error Evaluasi Test: {e}", np.array([])


def preprocess_single_comment(text_input):
    """
    Menjalankan pipeline preprocessing (6 langkah) pada satu string input.
    Wajib menggunakan resource (slang, stopwords) yang sama dengan training.
    """
    stop_list = get_stopwords_list() 
    slang_dict = SLANGWORDS 

    if stop_list is None:
        st.error("Gagal memuat stopwords untuk inferensi.")
        return None
        
    try:
        # Cleaning + Casefolding
        clean_text = cleaningText(text_input)
        # Fix slangwords
        slang_text = fix_slangwords(clean_text, slang_dict)
        # Tokenizing
        token_text = tokenizingText(slang_text)
        # Stopword removal
        stop_text = filteringText(token_text, stop_list)
        # Convert to sentence
        final_text = toSentence(stop_text)
        return final_text
    except LookupError as e: 
        st.error(f"Error NLTK saat preprocessing input: {e}. Pastikan data NLTK terunduh. Coba refresh halaman.")
        return None
    except Exception as e: 
        st.error(f"Error tidak terduga saat preprocessing input: {e}")
        return None


@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

# =======================================================================
# UI STREAMLIT
# =======================================================================

# Upload Data 
st.header("Upload Data")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

@st.cache_data(show_spinner=False) 
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        return None

if uploaded_file is not None:
    with st.spinner(f"Membaca file '{uploaded_file.name}'..."):
        df = load_data(uploaded_file)

    if df is not None: 
        # Bersihkan state jika file baru diupload
        if 'current_file_name' not in st.session_state or st.session_state['current_file_name'] != uploaded_file.name:
            # Hapus semua state proses sebelumnya
            keys_to_delete = ['processed_data', 'labeled_data', 'model_ready', 'original_column_name', 'processed_file_name']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['current_file_name'] = uploaded_file.name
            st.success(f"File '{uploaded_file.name}' berhasil dimuat. State sebelumnya dibersihkan.")

        # Preview data asli
        st.write("Preview Data Asli (5 baris):")
        st.dataframe(df.head())
        
        # Pilih Kolom & Preprocessing
        st.markdown("---")
        st.header("Langkah 1: Preprocessing Data")
        
        with st.expander("ℹ️ Proses Apa Saja yang Dilakukan?"):
             st.markdown("""
             Tahapan ini akan membersihkan teks komentar melalui langkah-langkah berikut:
             1.  **Cleaning:** Menghapus emoji, mention (@), hashtag (#), URL (http/www), angka, tanda baca, dan spasi berlebih.
             2.  **Case Folding:** Mengubah semua huruf menjadi huruf kecil (lowercase).
             3.  **Normalisasi Slang:** Mengganti kata-kata slang (tidak baku) menjadi kata baku (contoh: 'abis' -> 'habis').
             4.  **Tokenizing:** Memecah kalimat menjadi kata-kata (token).
             5.  **Stopword Removal:** Menghapus kata-kata umum yang tidak memiliki makna signifikan (contoh: 'yang', 'di', 'dan', 'sih').
             6.  **Convert to Sentence:** Menggabungkan kembali token menjadi kalimat utuh.
             """)
            
        column_to_process = st.selectbox(
            "Pilih kolom yang berisi teks komentar:",
             [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])],
            index=None,
            placeholder="Pilih kolom teks...",
            key="column_select" 
        )
        
        if column_to_process:
            if st.button("Mulai Proses Preprocessing", type="primary", key="preprocess_button", disabled=(not column_to_process)):

                # Validasi Data Input sebelum proses
                st.write("--- Validasi Data ---")
                validation_passed = True

                # Cek Stopwords
                if LIST_STOPWORDS is None:
                     st.error("❌ Daftar Stopwords gagal dimuat. Preprocessing tidak dapat dilanjutkan.")
                     validation_passed = False

                # Cek jumlah data
                min_rows_needed = 10
                if validation_passed and len(df) < min_rows_needed:
                    st.error(f"❌ Data terlalu sedikit! Minimal {min_rows_needed} baris diperlukan (ditemukan: {len(df)}).")
                    validation_passed = False

                # Cek kolom yang dipilih
                if validation_passed and column_to_process not in df.columns:
                     st.error(f"❌ Kolom '{column_to_process}' tidak ditemukan di data!")
                     validation_passed = False
                elif validation_passed and df[column_to_process].isna().all():
                    st.error(f"❌ Kolom '{column_to_process}' seluruhnya kosong (NaN/Null)! Pilih kolom lain.")
                    validation_passed = False

                # Cek tipe data (hanya warning)
                if validation_passed and not pd.api.types.is_string_dtype(df[column_to_process]) and not pd.api.types.is_object_dtype(df[column_to_process]):
                    st.warning(f"⚠️ Kolom '{column_to_process}' bukan tipe data teks. Akan dicoba konversi otomatis.")

                # Hitung null values
                if validation_passed:
                    null_count = df[column_to_process].isna().sum()
                    if null_count > 0:
                        st.info(f"ℹ️ Ditemukan {null_count} baris kosong (NaN/Null) di kolom '{column_to_process}'. Baris ini akan diabaikan.")

                st.write("--- Akhir Validasi ---")
                if not validation_passed:
                    st.stop() 

                with st.spinner("Sedang memproses data... Ini mungkin butuh waktu lama."):
                    # Terima DataFrame lengkap (unfiltered)
                    unfiltered_processed_df = run_full_preprocessing(df, column_to_process)
                
                # Cek hasil preprocessing
                if unfiltered_processed_df is None:
                    st.error("Preprocessing gagal. Silakan periksa log atau error di atas.")
                else:
                    st.success(f"Preprocessing Selesai! {len(unfiltered_processed_df)} baris telah diproses untuk file '{uploaded_file.name}'.")
                    
                    # preview dari data yang sudah diproses
                    st.write("Preview Data Hasil Preprocessing (5 baris):")
                    # kolom asli dan kolom baru yang relevan
                    preview_cols = [column_to_process, 'text_clean', 'text_slang', 'text_final', 'text_length']
                    st.dataframe(unfiltered_processed_df[preview_cols].head())
                    
                    # Simpan data ke session state
                    st.session_state['processed_data'] = unfiltered_processed_df
                    st.session_state['original_column_name'] = column_to_process
                    st.session_state['processed_file_name'] = uploaded_file.name
                    
                    # Reset state
                    if 'labeled_data' in st.session_state: del st.session_state['labeled_data']
                    if 'model_ready' in st.session_state: del st.session_state['model_ready']
                    
                    st.info("Data yang telah diproses disimpan. Lanjutkan ke Langkah 2 (Pelabelan).")

        # Pelabelan
        safe_to_label = False
        if 'processed_data' in st.session_state and st.session_state['processed_data'] is not None:
             if 'processed_file_name' in st.session_state and st.session_state['processed_file_name'] == uploaded_file.name:
                  safe_to_label = True

        if safe_to_label:
            st.markdown("---")
            st.header("Langkah 2: Pelabelan Otomatis (Lexicon)")
            
            st.info("""
            Tahap ini memberikan label sentimen (Positif/Negatif/Netral) secara otomatis pada teks bersih ('text_final').
            Metode yang digunakan adalah **Lexicon-Based** menggunakan kamus sentimen Bahasa Indonesia.
            """)
            st.markdown("""
            **Sumber Lexicon:** [Positif](https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv), [Negatif](https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv)
            """)
            
            label_choice = st.radio(
                "Pilih skema pelabelan sentimen:",
                ("3 Label (Positif, Neutral, Negatif)", "2 Label (Positif, Negatif)"),
                index=0, key="label_choice"
            )
            
            if st.button("Beri Label Sentimen", key="label_button"):
                num_labels = 3 if "3 Label" in label_choice else 2
                
                # Ambil data (unfiltered)
                df_to_label = st.session_state['processed_data']
                
                with st.spinner("Memuat lexicon..."):
                    lex_pos = load_lexicon_positive()
                    lex_neg = load_lexicon_negative()
                
                if lex_pos is not None and lex_neg is not None:
                    with st.spinner(f"Menjalankan pelabelan {num_labels} label pada {len(df_to_label)} baris..."):
                        df_labeled_unfiltered = run_labeling(df_to_label, 'text_final', num_labels, lex_pos, lex_neg)
                    
                    if df_labeled_unfiltered is None:
                         st.error("Proses pelabelan gagal.")
                    else:
                        # Lakukan filter
                        min_words = 2
                        df_labeled_filtered = df_labeled_unfiltered[df_labeled_unfiltered['text_length'] >= min_words].copy()

                        st.success(f"Pelabelan selesai! Ditemukan {len(df_labeled_filtered)} baris valid (>= {min_words} kata) untuk training.")
                        
                        # Tampilkan preview dari data
                        st.write("Preview Data Setelah Dilabeli (data yang akan dipakai model):")
                        st.dataframe(df_labeled_filtered[['text_final', 'polarity_score', 'sentiment']].head())
                        
                        # Simpan data ke session state
                        st.session_state['labeled_data'] = df_labeled_filtered
                        if 'model_ready' in st.session_state: del st.session_state['model_ready']
                        st.info("Data berlabel (yang telah difilter) telah disimpan. Lanjutkan ke Langkah 3.")
                        
                        # Tambahkan tombol download
                        csv_data = convert_df_to_csv(df_labeled_unfiltered)
                        st.download_button(
                            label="📥 Download Data Lengkap Hasil Labeling (.csv)",
                            data=csv_data,
                            file_name=f"labeled_{st.session_state.get('processed_file_name', 'data')}.csv",
                            mime="text/csv",
                            key="download_csv_labeled"
                        )
                else:
                    st.error("Pelabelan tidak dapat dilanjutkan karena lexicon gagal dimuat.")

        # Visualisasi
        safe_to_visualize = safe_to_label and 'labeled_data' in st.session_state and st.session_state['labeled_data'] is not None
        if safe_to_visualize:
            st.markdown("---")
            st.header("Langkah 3: Visualisasi")
            
            df_labeled_viz = st.session_state['labeled_data'] 
            if df_labeled_viz.empty:
                st.warning("Tidak ada data valid yang tersisa setelah filter untuk divisualisasikan.")
            else:
                # Visualisasi Pie Plot 
                st.subheader("Distribusi Sentimen")
                sentiment_counts = df_labeled_viz['sentiment'].value_counts()
                if sentiment_counts.empty:
                    st.warning("Tidak ada data sentimen untuk Pie Plot.")
                else:
                    labels = sentiment_counts.index
                    sizes = sentiment_counts.values
                    def make_autopct(values):
                        def my_autopct(pct):
                            total = sum(values)
                            val = int(round(pct * total / 100.0))
                            if total == 0: return f'0.0%\n(0 data)'
                            return f'{pct:.0f}%\n({val} data)'
                        return my_autopct
                        
                    col1, col2 = st.columns([1, 1]) 
                    
                    with col1:
                        fig_pie, ax_pie = plt.subplots(figsize=(6, 5), dpi=100)
                        
                        ax_pie.pie(
                            sizes, 
                            labels=labels, 
                            autopct=make_autopct(sizes),
                            startangle=140, 
                            colors=sns.color_palette('pastel'),
                            wedgeprops={'edgecolor': 'black'},
                            textprops={'fontsize': 9}
                        )
                        
                        ax_pie.set_title('Distribusi Sentimen', fontsize=12)
                        ax_pie.axis('equal')
                        
                        plt.tight_layout()
                        st.pyplot(fig_pie)
                        plt.close(fig_pie)
                    
                    with col2:
                        st.write("") 
                
                # Visualisasi Word Cloud 
                st.subheader("Word Cloud per Sentimen")
                sentiments = df_labeled_viz['sentiment'].unique()
                if len(sentiments) == 0:
                     st.warning("Tidak ada sentimen unik untuk Word Cloud.")
                else:
                    num_sentiments = len(sentiments)
                    if num_sentiments > 0:
                        fig_wc, axes_wc = plt.subplots(1, num_sentiments, figsize=(7 * num_sentiments, 7), squeeze=False)
                        axes_wc_flat = axes_wc.flatten()

                        for i, sentiment in enumerate(sentiments):
                            ax = axes_wc_flat[i]
                            text_corpus = " ".join(df_labeled_viz[df_labeled_viz['sentiment'] == sentiment]['text_final'].astype(str))
                            if not text_corpus.strip():
                                ax.set_title(f"Word Cloud - {sentiment}\n(Tidak ada data)", fontsize=16)
                                ax.axis('off')
                                continue
                            try:
                                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                                      colormap='viridis', max_words=100,
                                                      collocation_threshold=30).generate(text_corpus)
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.set_title(f"Word Cloud - Sentimen '{sentiment}'", fontsize=16)
                            except ValueError as e:
                                 st.warning(f"Tidak dapat membuat Word Cloud untuk '{sentiment}': {e}")
                                 ax.set_title(f"Word Cloud - {sentiment}\n(Error: {e})", fontsize=12)
                            ax.axis('off')
                        plt.tight_layout()
                        st.pyplot(fig_wc)

        # Modelling
        safe_to_model = safe_to_label and 'labeled_data' in st.session_state and st.session_state['labeled_data'] is not None
        if safe_to_model:
            st.markdown("---")
            st.header("Langkah 4: Modelling")
            
            st.subheader("Pengaturan Model")
            
            # Pilihan Ekstraksi Fitur 
            feature_choice = st.radio(
                "Pilih Metode Ekstraksi Fitur:",
                ("TF-IDF", "Bag-of-Words (BoW)"),
                key="feature_choice"
            )
            
            # Pilihan Parameter Fitur
            feature_param_labels = [
                "Default (Seimbang: max_features=5000, min_df=5, max_df=0.8)",
                "Terfokus (Focused: max_features=3000, min_df=10, max_df=0.7)",
                "Luas (Broad: max_features=None, min_df=2, max_df=0.95)"
            ]
            feature_param_values = [
                {'max_features': 5000, 'min_df': 5, 'max_df': 0.8},
                {'max_features': 3000, 'min_df': 10, 'max_df': 0.7},
                {'max_features': None, 'min_df': 2, 'max_df': 0.95}
            ]
            
            selected_feature_label = st.selectbox(
                f"Pilih Set Parameter untuk {feature_choice}:",
                options=feature_param_labels,
                index=0, 
                key="feature_params"
            )
            feature_params = feature_param_values[feature_param_labels.index(selected_feature_label)]

            # Pilihan Model
            model_choice = st.radio(
                "Pilih Model Klasifikasi:",
                ("Logistic Regression", "SVM (Support Vector Machine)", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"),
                key="model_choice"
            )
            
            # Blok Parameter Kondisional Model
            model_params = {} 
            st.subheader(f"Parameter untuk {model_choice}")

            # Label opsi C
            c_labels = ["0.01", "0.1", "1.0 (Default)", "10.0", "100.0"]
            c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
            c_default_index = 2 

            # Label opsi Alpha
            alpha_labels = ["0.01", "0.1", "0.5", "1.0 (Default)"]
            alpha_values = [0.01, 0.1, 0.5, 1.0]
            alpha_default_index = 3 

            # Label opsi Max Iter
            max_iter_labels = ["100", "500", "1000 (Default)", "2000", "5000"]
            max_iter_values = [100, 500, 1000, 2000, 5000]
            max_iter_default_index = 2 

            if model_choice == "Logistic Regression":
                # Selectbox untuk 'C'
                selected_c_label = st.selectbox(
                    "Parameter 'C' (Kekuatan Regularisasi)", 
                    options=c_labels, 
                    index=c_default_index, 
                    key="C_lr"
                )
                model_params['C_lr'] = c_values[c_labels.index(selected_c_label)]
                
                # Selectbox untuk 'max_iter'
                selected_max_iter_label = st.selectbox(
                    "Max Iterations (untuk konvergensi)",
                    options=max_iter_labels,
                    index=max_iter_default_index,
                    key="max_iter_lr"
                )
                model_params['max_iter_lr'] = max_iter_values[max_iter_labels.index(selected_max_iter_label)]

            elif model_choice == "SVM (Support Vector Machine)":
                # Selectbox untuk 'C'
                selected_c_label = st.selectbox(
                    "Parameter 'C' (Kekuatan Regularisasi)", 
                    options=c_labels, 
                    index=c_default_index, 
                    key="C_svm"
                )
                model_params['C_svm'] = c_values[c_labels.index(selected_c_label)]
                model_params['kernel_svm'] = st.radio("Pilih Kernel:", ('rbf', 'linear', 'poly'), key="kernel_svm")

            elif model_choice == "Multinomial Naive Bayes":
                # Selectbox untuk 'alpha'
                selected_alpha_label = st.selectbox(
                    "Parameter 'alpha' (Laplace Smoothing)", 
                    options=alpha_labels, 
                    index=alpha_default_index, 
                    key="alpha_mnb"
                )
                model_params['alpha_mnb'] = alpha_values[alpha_labels.index(selected_alpha_label)]

            elif model_choice == "Bernoulli Naive Bayes":
                # Selectbox untuk 'alpha'
                selected_alpha_label = st.selectbox(
                    "Parameter 'alpha' (Laplace Smoothing)", 
                    options=alpha_labels, 
                    index=alpha_default_index, 
                    key="alpha_bnb"
                )
                model_params['alpha_bnb'] = alpha_values[alpha_labels.index(selected_alpha_label)]
            
            if st.button("Latih Model", type="primary"):
                 # Ambil data yang sudah difilter dan dilabeli
                df_to_train = st.session_state.get('labeled_data') 

                # Validasi tambahan sebelum training
                if df_to_train is None or df_to_train.empty:
                    st.error("Tidak ada data valid yang tersedia untuk melatih model.")
                else:
                    with st.spinner(f"Melatih model {model_choice} dengan {feature_choice}..."):
                        # Panggil fungsi training
                        model, vectorizer, le, accuracy_train, accuracy_test, report, cm = train_model(
                            df_to_train,
                            'text_final',
                            'sentiment',
                            feature_choice,
                            feature_params,
                            model_choice,
                            model_params
                        )

                    # Cek jika training gagal
                    if model is None:
                         st.error("Pelatihan model gagal. Silakan periksa error di atas.")
                    else:
                        st.success("Pelatihan model selesai!")

                        # Simpan artefak penting ke session state untuk inferensi
                        st.session_state['model'] = model
                        st.session_state['vectorizer'] = vectorizer
                        st.session_state['label_encoder'] = le
                        st.session_state['model_ready'] = True

                        # Tampilkan Hasil Evaluasi
                        st.subheader("Hasil Evaluasi Model")

                        # Akurasi
                        st.metric(label="Akurasi Model (pada Data Train)", value=f"{accuracy_train * 100:.2f}%")
                        st.metric(label="Akurasi Model (pada Data Test)", value=f"{accuracy_test * 100:.2f}%")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Confusion Matrix")
                            if cm.size > 0 and hasattr(le, 'classes_'):
                                fig_cm, ax_cm = plt.subplots(figsize=(5, 4), dpi=100)
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_xlabel('Predicted')
                                plt.tight_layout()
                                st.pyplot(fig_cm)
                                plt.close(fig_cm)
                        
                        with col2:
                            st.subheader("Classification Report")
                            st.text(report)


                        st.info("Model, Vectorizer, dan Label Encoder telah disimpan. Lanjutkan ke Langkah 5.")

                        # Blok Download Model (ZIP)
                        st.subheader("Download Artefak Model")
                        st.warning("Untuk menggunakan model ini di aplikasi lain, memerlukan file ZIP ini DAN **skrip preprocessing yang identik**.")

                        try:
                            # Buat file ZIP
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                                zip_file.writestr("sentiment_model.pkl", pickle.dumps(st.session_state['model']))
                                zip_file.writestr("vectorizer.pkl", pickle.dumps(st.session_state['vectorizer']))
                                zip_file.writestr("label_encoder.pkl", pickle.dumps(st.session_state['label_encoder']))

                            # Siapkan data untuk tombol download
                            zip_data = zip_buffer.getvalue()

                            # Tombol download ZIP
                            st.download_button(
                                label="📥 Download Artefak Model (.zip)",
                                data=zip_data,
                                file_name="sentiment_model_artifacts.zip",
                                mime="application/zip",
                                key="download_zip"
                            )
                        except Exception as e:
                            st.error(f"Gagal membuat file ZIP: {e}")

        # Inferensi
        safe_to_infer = safe_to_model and 'model_ready' in st.session_state and st.session_state['model_ready']
        if safe_to_infer:
            st.markdown("---")
            st.header("Langkah 5: Coba Prediksi (Inferensi)")
            
            new_comment = st.text_area("Masukkan komentar baru untuk diprediksi:", key="new_comment_input")
            
            if st.button("Prediksi Sentimen", key="predict_button"):
                if not new_comment.strip():
                    st.warning("Silakan masukkan komentar terlebih dahulu.")
                else:
                    if not nltk_setup_successful:
                         st.error("Tidak dapat melakukan prediksi karena setup NLTK gagal.")
                    elif 'model' not in st.session_state or 'vectorizer' not in st.session_state or 'label_encoder' not in st.session_state:
                         st.error("Artefak model tidak ditemukan di session state. Harap latih ulang model.")
                    else:
                        with st.spinner("Memproses dan memprediksi..."):
                            try:
                                # Ambil artefak dari session state
                                model = st.session_state['model']
                                vectorizer = st.session_state['vectorizer']
                                le = st.session_state['label_encoder']
                                
                                # Preprocessing teks input
                                processed_text = preprocess_single_comment(new_comment)
                                
                                if processed_text is None:
                                     st.error("Preprocessing teks input gagal.")
                                elif not processed_text.strip():
                                     st.warning("Teks input menjadi kosong setelah preprocessing. Tidak dapat diprediksi.")
                                else:
                                    # Vectorize
                                    vectorized_text = vectorizer.transform([processed_text])
                                    
                                    # Predict
                                    prediction_numeric = model.predict(vectorized_text)
                                    
                                    # Decode label
                                    prediction_label = le.inverse_transform(prediction_numeric)
                                    
                                    # Tampilkan hasil
                                    st.subheader("Hasil Prediksi:")
                                    st.success(f"Sentimen: **{prediction_label[0]}**")
                                    
                                    # Tampilkan detail
                                    with st.expander("Lihat detail proses"):
                                        st.write(f"**Teks Asli:**")
                                        st.write(new_comment)
                                        st.write(f"**Teks Bersih (input model):**")
                                        st.write(processed_text)
                                
                            except Exception as e:
                                st.error(f"Terjadi error saat prediksi: {e}")
                                st.exception(e)


else:
    if nltk_setup_successful: 
         st.info("Silakan upload file CSV untuk memulai.")
    else:
         st.error("Setup NLTK Gagal. Harap periksa koneksi internet atau coba refresh halaman.")
