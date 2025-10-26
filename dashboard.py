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
st.title("Pipeline Analisis Sentimen Interaktif 1.0")
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
SLANGWORDS = {"@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah", "afaik": "as far as i know", "ahaha": "haha", "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku", "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur", "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik", "asekk": "asyik", "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bandes": "bantuan desa", "bangedh": "banget", "banpol": "bantuan polisi", "banpur": "bantuan tempur", "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "begajulan": "nakal", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga", "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana", "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar", "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng", "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia", "bubbu": "tidur", "bubu": "tidur", "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "calo": "makelar", "can": "belum", "capcus": "pergi", "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja", "cipika cipiki": "cium pipi kanan cium pipi kiri", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep", "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci", "conlok": "cinta lokasi", "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok", "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "curanmor": "pencurian kendaraan bermotor", "curcol": "curahan hati colongan", "cwek": "cewek", "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan", "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan", "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian", "ed": "edisi", "egp": "emang gue pikirin", "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile", "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan", "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja", "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas", "gepeng": "gelandangan dan pengemis", "ghiy": "lagi", "gile": "gila", "gimana": "bagaimana", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas", "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "golput": "golongan putih", "gowes": "mengayuh sepeda", "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu", "gua": "saya", "guoblok": "goblok", "gw": "saya", "ha": "tertawa", "haha": "tertawa", "hallow": "halo", "hankam": "pertahanan dan keamanan", "hehe": "he", "helo": "halo", "hey": "hai", "hlm": "halaman", "hny": "hanya", "hoax": "isu bohong", "hr": "hari", "hrus": "harus", "hubdar": "perhubungan darat", "huff": "mengeluh", "hum": "rumah", "humz": "rumah", "ilang": "hilang", "ilfil": "tidak suka", "imho": "in my humble opinion", "imoetz": "imut", "item": "hitam", "itungan": "hitungan", "iye": "iya", "ja": "saja", "jadiin": "jadi", "jaim": "jaga image", "jayus": "tidak lucu", "jdi": "jadi", "jem": "jam", "jga": "juga", "jgnkan": "jangankan", "jir": "anjing", "jln": "jalan", "jomblo": "tidak punya pacar", "jubir": "juru bicara", "jutek": "galak", "k": "ke", "kab": "kabupaten", "kabor": "kabur", "kacrut": "kacau", "kadiv": "kepala divisi", "kagak": "tidak", "kalo": "kalau", "kampret": "sialan", "kamtibmas": "keamanan dan ketertiban masyarakat", "kamuwh": "kamu", "kanwil": "kantor wilayah", "karna": "karena", "kasubbag": "kepala subbagian", "katrok": "kampungan", "kayanya": "kayaknya", "kbr": "kabar", "kdu": "harus", "kec": "kecamatan", "kejurnas": "kejuaraan nasional", "kekeuh": "keras kepala", "kel": "kelurahan", "kemaren": "kemarin", "kepengen": "mau", "kepingin": "mau", "kepsek": "kepala sekolah", "kesbang": "kesatuan bangsa", "kesra": "kesejahteraan rakyat", "ketrima": "diterima", "kgiatan": "kegiatan", "kibul": "bohong", "kimpoi": "kawin", "kl": "kalau", "klianz": "kalian", "kloter": "kelompok terbang", "klw": "kalau", "km": "kamu", "kmps": "kampus", "kmrn": "kemarin", "knal": "kenal", "knp": "kenapa", "kodya": "kota madya", "komdis": "komisi disiplin", "komsov": "komunis sovyet", "kongkow": "kumpul bareng teman-teman", "kopdar": "kopi darat", "korup": "korupsi", "kpn": "kapan", "krenz": "keren", "krm": "kirim", "kt": "kita", "ktmu": "ketemu", "ktr": "kantor", "kuper": "kurang pergaulan", "kw": "imitasi", "kyk": "seperti", "la": "lah", "lam": "salam", "lamp": "lampiran", "lanud": "landasan udara", "latgab": "latihan gabungan", "lebay": "berlebihan", "leh": "boleh", "lelet": "lambat", "lemot": "lambat", "lgi": "lagi", "lgsg": "langsung", "liat": "lihat", "litbang": "penelitian dan pengembangan", "lmyn": "lumayan", "lo": "kamu", "loe": "kamu", "lola": "lambat berfikir", "louph": "cinta", "low": "kalau", "lp": "lupa", "luber": "langsung, umum, bebas, dan rahasia", "luchuw": "lucu", "lum": "belum", "luthu": "lucu", "lwn": "lawan", "maacih": "terima kasih", "mabal": "bolos", "macem": "macam", "macih": "masih", "maem": "makan", "magabut": "makan gaji buta", "maho": "homo", "mak jang": "kaget", "maksain": "memaksa", "malem": "malam", "mam": "makan", "maneh": "kamu", "maniez": "manis", "mao": "mau", "masukin": "masukkan", "melu": "ikut", "mepet": "dekat sekali", "mgu": "minggu", "migas": "minyak dan gas bumi", "mikol": "minuman beralkohol", "miras": "minuman keras", "mlah": "malah", "mngkn": "mungkin", "mo": "mau", "mokad": "mati", "moso": "masa", "mpe": "sampai", "msk": "masuk", "mslh": "masalah", "mt": "makan teman", "mubes": "musyawarah besar", "mulu": "melulu", "mumpung": "selagi", "munas": "musyawarah nasional", "muntaber": "muntah dan berak", "musti": "mesti", "muupz": "maaf", "mw": "now watching", "n": "dan", "nanam": "menanam", "nanya": "bertanya", "napa": "kenapa", "napi": "narapidana", "napza": "narkotika, alkohol, psikotropika, dan zat adiktif ", "narkoba": "narkotika, psikotropika, dan obat terlarang", "nasgor": "nasi goreng", "nda": "tidak", "ndiri": "sendiri", "ne": "ini", "nekolin": "neokolonialisme", "nembak": "menyatakan cinta", "ngabuburit": "menunggu berbuka puasa", "ngaku": "mengaku", "ngambil": "mengambil", "nganggur": "tidak punya pekerjaan", "ngapah": "kenapa", "ngaret": "terlambat", "ngasih": "memberikan", "ngebandel": "berbuat bandel", "ngegosip": "bergosip", "ngeklaim": "mengklaim", "ngeksis": "menjadi eksis", "ngeles": "berkilah", "ngelidur": "menggigau", "ngerampok": "merampok", "ngga": "tidak", "ngibul": "berbohong", "ngiler": "mau", "ngiri": "iri", "ngisiin": "mengisikan", "ngmng": "bicara", "ngomong": "bicara", "ngubek2": "mencari-cari", "ngurus": "mengurus", "nie": "ini", "nih": "ini", "niyh": "nih", "nmr": "nomor", "nntn": "nonton", "nobar": "nonton bareng", "np": "now playing", "ntar": "nanti", "ntn": "nonton", "numpuk": "bertumpuk", "nutupin": "menutupi", "nyari": "mencari", "nyekar": "menyekar", "nyicil": "mencicil", "nyoblos": "mencoblos", "nyokap": "ibu", "ogah": "tidak mau", "ol": "online", "ongkir": "ongkos kirim", "oot": "out of topic", "org2": "orang-orang", "ortu": "orang tua", "otda": "otonomi daerah", "otw": "on the way, sedang di jalan", "pacal": "pacar", "pake": "pakai", "pala": "kepala", "pansus": "panitia khusus", "parpol": "partai politik", "pasutri": "pasangan suami istri", "pd": "pada", "pede": "percaya diri", "pelatnas": "pemusatan latihan nasional", "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", "pemred": "pemimpin redaksi", "penjas": "pendidikan jasmani", "perda": "peraturan daerah", "perhatiin": "perhatikan", "pesenan": "pesanan", "pgang": "pegang", "pi": "tapi", "pilkada": "pemilihan kepala daerah", "pisan": "sangat", "pk": "penjahat kelamin", "plg": "paling", "pmrnth": "pemerintah", "polantas": "polisi lalu lintas", "ponpes": "pondok pesantren", "pp": "pulang pergi", "prg": "pergi", "prnh": "pernah", "psen": "pesan", "pst": "pasti", "pswt": "pesawat", "pw": "posisi nyaman", "qmu": "kamu", "rakor": "rapat koordinasi", "ranmor": "kendaraan bermotor", "re": "reply", "ref": "referensi", "rehab": "rehabilitasi", "rempong": "sulit", "repp": "balas", "restik": "reserse narkotika", "rhs": "rahasia", "rmh": "rumah", "ru": "baru", "ruko": "rumah toko", "rusunawa": "rumah susun sewa", "ruz": "terus", "saia": "saya", "salting": "salah tingkah", "sampe": "sampai", "samsek": "sama sekali", "sapose": "siapa", "satpam": "satuan pengamanan", "sbb": "sebagai berikut", "sbh": "sebuah", "sbnrny": "sebenarnya", "scr": "secara", "sdgkn": "sedangkan", "sdkt": "sedikit", "se7": "setuju", "sebelas dua belas": "mirip", "sembako": "sembilan bahan pokok", "sempet": "sempat", "sendratari": "seni drama tari", "sgt": "sangat", "shg": "sehingga", "siech": "sih", "sikon": "situasi dan kondisi", "sinetron": "sinema elektronik", "siramin": "siramkan", "sj": "saja", "skalian": "sekalian", "sklh": "sekolah", "skt": "sakit", "slesai": "selesai", "sll": "selalu", "slma": "selama", "slsai": "selesai", "smpt": "sempat", "smw": "semua", "sndiri": "sendiri", "soljum": "sholat jumat", "songong": "sombong", "sory": "maaf", "sosek": "sosial-ekonomi", "sotoy": "sok tahu", "spa": "siapa", "sppa": "siapa", "spt": "seperti", "srtfkt": "sertifikat", "stiap": "setiap", "stlh": "setelah", "suk": "masuk", "sumpek": "sempit", "syg": "sayang", "t4": "tempat", "tajir": "kaya", "tau": "tahu", "taw": "tahu", "td": "tadi", "tdk": "tidak", "teh": "kakak perempuan", "telat": "terlambat", "telmi": "telat berpikir", "temen": "teman", "tengil": "menyebalkan", "tepar": "terkapar", "tggu": "tunggu", "tgu": "tunggu", "thankz": "terima kasih", "thn": "tahun", "tilang": "bukti pelanggaran", "tipiwan": "TvOne", "tks": "terima kasih", "tlp": "telepon", "tls": "tulis", "tmbah": "tambah", "tmen2": "teman-teman", "tmpah": "tumpah", "tmpt": "tempat", "tngu": "tunggu", "tnyta": "ternyata", "tokai": "tai", "toserba": "toko serba ada", "tpi": "tapi", "trdhulu": "terdahulu", "trima": "terima kasih", "trm": "terima", "trs": "terus", "trutama": "terutama", "ts": "penulis", "tst": "tahu sama tahu", "ttg": "tentang", "tuch": "tuh", "tuir": "tua", "tw": "tahu", "u": "kamu", "ud": "sudah", "udah": "sudah", "ujg": "ujung", "ul": "ulangan", "unyu": "lucu", "uplot": "unggah", "urang": "saya", "usah": "perlu", "utk": "untuk", "valas": "valuta asing", "w/": "dengan", "wadir": "wakil direktur", "wamil": "wajib militer", "warkop": "warung kopi", "warteg": "warung tegal", "wat": "buat", "wkt": "waktu", "wtf": "what the fuck", "xixixi": "tertawa", "ya": "iya", "yap": "iya", "yaudah": "ya sudah", "yawdah": "ya sudah", "yg": "yang", "yl": "yang lain", "yo": "iya", "yowes": "ya sudah", "yup": "iya", "7an": "tujuan", "ababil": "abg labil", "acc": "accord", "adlah": "adalah", "adoh": "aduh", "aha": "tertawa", "aing": "saya", "aja": "saja", "ajj": "saja", "aka": "dikenal juga sebagai", "akko": "aku", "akku": "aku", "akyu": "aku", "aljasa": "asal jadi saja", "ama": "sama", "ambl": "ambil", "anjir": "anjing", "ank": "anak", "ap": "apa", "apaan": "apa", "ape": "apa", "aplot": "unggah", "apva": "apa", "aqu": "aku", "asap": "sesegera mungkin", "aseek": "asyik", "asek": "asyik", "aseknya": "asyiknya", "asoy": "asyik", "astrojim": "astagfirullahaladzim", "ath": "kalau begitu", "atuh": "kalau begitu", "ava": "avatar", "aws": "awas", "ayang": "sayang", "ayok": "ayo", "bacot": "banyak bicara", "bales": "balas", "bangdes": "pembangunan desa", "bangkotan": "tua", "banpres": "bantuan presiden", "bansarkas": "bantuan sarana kesehatan", "bazis": "badan amal, zakat, infak, dan sedekah", "bcoz": "karena", "beb": "sayang", "bejibun": "banyak", "belom": "belum", "bener": "benar", "ber2": "berdua", "berdikari": "berdiri di atas kaki sendiri", "bet": "banget", "beti": "beda tipis", "beut": "banget", "bgd": "banget", "bgs": "bagus", "bhubu": "tidur", "bimbuluh": "bimbingan dan penyuluhan", "bisi": "kalau-kalau", "bkn": "bukan", "bl": "beli", "blg": "bilang", "blm": "belum", "bls": "balas", "bnchi": "benci", "bngung": "bingung", "bnyk": "banyak", "bohay": "badan aduhai", "bokep": "porno", "bokin": "pacar", "bole": "boleh", "bolot": "bodoh", "bonyok": "ayah ibu", "bpk": "bapak", "brb": "segera kembali", "brngkt": "berangkat", "brp": "berapa", "brur": "saudara laki-laki", "bsa": "bisa", "bsk": "besok", "bu_bu": "tidur", "bubarin": "bubarkan", "buber": "buka bersama", "bujubune": "luar biasa", "buser": "buru sergap", "bwhn": "bawahan", "byar": "bayar", "byr": "bayar", "c8": "chat", "cabut": "pergi", "caem": "cakep", "cama-cama": "sama-sama", "cangcut": "celana dalam", "cape": "capek", "caur": "jelek", "cekak": "tidak ada uang", "cekidot": "coba lihat", "cemplungin": "cemplungkan", "ceper": "pendek", "ceu": "kakak perempuan", "cewe": "cewek", "cibuk": "sibuk", "cin": "cinta", "ciye": "cie", "ckck": "ck", "clbk": "cinta lama bersemi kembali", "cmpr": "campur", "cnenk": "senang", "congor": "mulut", "cow": "cowok", "coz": "karena", "cpa": "siapa", "gokil": "gila", "gombal": "suka merayu", "gpl": "tidak pakai lama", "gpp": "tidak apa-apa", "gretong": "gratis", "gt": "begitu", "gtw": "tidak tahu", "gue": "saya", "guys": "teman-teman", "gws": "cepat sembuh", "haghaghag": "tertawa", "hakhak": "tertawa", "handak": "bahan peledak", "hansip": "pertahanan sipil", "hellow": "halo", "helow": "halo", "hi": "hai", "hlng": "hilang", "hnya": "hanya", "houm": "rumah", "hrs": "harus", "hubad": "hubungan angkatan darat", "hubla": "perhubungan laut", "huft": "mengeluh", "humas": "hubungan masyarakat", "idk": "saya tidak tahu", "ilfeel": "tidak suka", "imba": "jago sekali", "imoet": "imut", "info": "informasi", "itung": "hitung", "isengin": "bercanda", "iyala": "iya lah", "iyo": "iya", "jablay": "jarang dibelai", "jadul": "jaman dulu", "jancuk": "anjing", "jd": "jadi", "jdikan": "jadikan", "jg": "juga", "jgn": "jangan", "jijay": "jijik", "jkt": "jakarta", "jnj": "janji", "jth": "jatuh", "jurdil": "jujur adil", "jwb": "jawab", "ka": "kakak", "kabag": "kepala bagian", "kacian": "kasihan", "kadit": "kepala direktorat", "kaga": "tidak", "kaka": "kakak", "kamtib": "keamanan dan ketertiban", "kamuh": "kamu", "kamyu": "kamu", "kapt": "kapten", "kasat": "kepala satuan", "kasubbid": "kepala subbidang", "kau": "kamu", "kbar": "kabar", "kcian": "kasihan", "keburu": "terlanjur", "kedubes": "kedutaan besar", "kek": "seperti", "keknya": "kayaknya", "keliatan": "kelihatan", "keneh": "masih", "kepikiran": "terpikirkan", "kepo": "mau tahu urusan orang", "kere": "tidak punya uang", "kesian": "kasihan", "ketauan": "ketahuan", "keukeuh": "keras kepala", "khan": "kan", "kibus": "kaki busuk", "kk": "kakak", "klian": "kalian", "klo": "kalau", "kluarga": "keluarga", "klwrga": "keluarga", "kmari": "kemari", "kmpus": "kampus", "kn": "kan", "knl": "kenal", "knpa": "kenapa", "kog": "kok", "kompi": "komputer", "komtiong": "komunis Tiongkok", "konjen": "konsulat jenderal", "koq": "kok", "kpd": "kepada", "kptsan": "keputusan", "krik": "garing", "krn": "karena", "ktauan": "ketahuan", "ktny": "katanya", "kudu": "harus", "kuq": "kok", "ky": "seperti", "kykny": "kayanya", "laka": "kecelakaan", "lambreta": "lambat", "lansia": "lanjut usia", "lapas": "lembaga pemasyarakatan", "lbur": "libur", "lekong": "laki-laki", "lg": "lagi", "lgkp": "lengkap", "lht": "lihat", "linmas": "perlindungan masyarakat", "lmyan": "lumayan", "lngkp": "lengkap", "loch": "loh", "lol": "tertawa", "lom": "belum", "loupz": "cinta", "lowh": "kamu", "lu": "kamu", "luchu": "lucu", "luff": "cinta", "luph": "cinta", "lw": "kamu", "lwt": "lewat", "maaciw": "terima kasih", "mabes": "markas besar", "macem-macem": "macam-macam", "madesu": "masa depan suram", "maen": "main", "mahatma": "maju sehat bersama", "mak": "ibu", "makasih": "terima kasih", "malah": "bahkan", "malu2in": "memalukan", "mamz": "makan", "manies": "manis", "mantep": "mantap", "markus": "makelar kasus", "mba": "mbak", "mending": "lebih baik", "mgkn": "mungkin", "mhn": "mohon", "miker": "minuman keras", "milis": "mailing list", "mksd": "maksud", "mls": "malas", "mnt": "minta", "moge": "motor gede", "mokat": "mati", "mosok": "masa", "msh": "masih", "mskpn": "meskipun", "msng2": "masing-masing", "muahal": "mahal", "muker": "musyawarah kerja", "mumet": "pusing", "muna": "munafik", "munaslub": "musyawarah nasional luar biasa", "musda": "musyawarah daerah", "muup": "maaf", "muuv": "maaf", "nal": "kenal", "nangis": "menangis", "naon": "apa", "napol": "narapidana politik", "naq": "anak", "narsis": "bangga pada diri sendiri", "nax": "anak", "ndak": "tidak", "ndut": "gendut", "nekolim": "neokolonialisme", "nelfon": "menelepon", "ngabis2in": "menghabiskan", "ngakak": "tertawa", "ngambek": "marah", "ngampus": "pergi ke kampus", "ngantri": "mengantri", "ngapain": "sedang apa", "ngaruh": "berpengaruh", "ngawur": "berbicara sembarangan", "ngeceng": "kumpul bareng-bareng", "ngeh": "sadar", "ngekos": "tinggal di kos", "ngelamar": "melamar", "ngeliat": "melihat", "ngemeng": "bicara terus-terusan", "ngerti": "mengerti", "nggak": "tidak", "ngikut": "ikut", "nginep": "menginap", "ngisi": "mengisi", "ngmg": "bicara", "ngocol": "lucu", "ngomongin": "membicarakan", "ngumpul": "berkumpul", "ni": "ini", "nyasar": "tersesat", "nyariin": "mencari", "nyiapin": "mempersiapkan", "nyiram": "menyiram", "nyok": "ayo", "o/": "oleh", "ok": "ok", "priksa": "periksa", "pro": "profesional", "psn": "pesan", "psti": "pasti", "puanas": "panas", "qmo": "kamu", "qt": "kita", "rame": "ramai", "raskin": "rakyat miskin", "red": "redaksi", "reg": "register", "rejeki": "rezeki", "renstra": "rencana strategis", "reskrim": "reserse kriminal", "sni": "sini", "somse": "sombong sekali", "sorry": "maaf", "sosbud": "sosial-budaya", "sospol": "sosial-politik", "sowry": "maaf", "spd": "sepeda", "sprti": "seperti", "spy": "supaya", "stelah": "setelah", "subbag": "subbagian", "sumbangin": "sumbangkan", "sy": "saya", "syp": "siapa", "tabanas": "tabungan pembangunan nasional", "tar": "nanti", "taun": "tahun", "tawh": "tahu", "tdi": "tadi", "te2p": "tetap", "tekor": "rugi", "telkom": "telekomunikasi", "telp": "telepon", "temen2": "teman-teman", "tengok": "menjenguk", "terbitin": "terbitkan", "tgl": "tanggal", "thanks": "terima kasih", "thd": "terhadap", "thx": "terima kasih", "tipi": "TV", "tkg": "tukang", "tll": "terlalu", "tlpn": "telepon", "tman": "teman", "tmbh": "tambah", "tmn2": "teman-teman", "tmph": "tumpah", "tnda": "tanda", "tnh": "tanah", "togel": "toto gelap", "tp": "tapi", "tq": "terima kasih", "trgntg": "tergantung", "trims": "terima kasih", "cb": "coba", "y": "ya", "munfik": "munafik", "reklamuk": "reklamasi", "sma": "sama", "tren": "trend", "ngehe": "kesal", "mz": "mas", "analisise": "analisis", "sadaar": "sadar", "sept": "september", "nmenarik": "menarik", "zonk": "bodoh", "rights": "benar", "simiskin": "miskin", "ngumpet": "sembunyi", "hardcore": "keras", "akhirx": "akhirnya", "solve": "solusi", "watuk": "batuk", "ngebully": "intimidasi", "masy": "masyarakat", "still": "masih", "tauk": "tahu", "mbual": "bual", "tioghoa": "tionghoa", "ngentotin": "senggama", "kentot": "senggama", "faktakta": "fakta", "sohib": "teman", "rubahnn": "rubah", "trlalu": "terlalu", "nyela": "cela", "heters": "pembenci", "nyembah": "sembah", "most": "paling", "ikon": "lambang", "light": "terang", "pndukung": "pendukung", "setting": "atur", "seting": "akting", "next": "lanjut", "waspadalah": "waspada", "gantengsaya": "ganteng", "parte": "partai", "nyerang": "serang", "nipu": "tipu", "ktipu": "tipu", "jentelmen": "berani", "buangbuang": "buang", "tsangka": "tersangka", "kurng": "kurang", "ista": "nista", "less": "kurang", "koar": "teriak", "paranoid": "takut", "problem": "masalah", "tahi": "kotoran", "tirani": "tiran", "tilep": "tilap", "happy": "bahagia", "tak": "tidak", "penertiban": "tertib", "uasai": "kuasa", "mnolak": "tolak", "trending": "trend", "taik": "tahi", "wkwkkw": "tertawa", "ahokncc": "ahok", "istaa": "nista", "benarjujur": "jujur", "mgkin": "mungkin", "sya": "saya", "penipu²": "penipu", "teman²": "teman"}

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


