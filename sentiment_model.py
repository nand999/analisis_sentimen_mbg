import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download NLTK requirements
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        # Inisialisasi stemmer bahasa Indonesia
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Inisialisasi stopwords bahasa Indonesia
        stop_factory = StopWordRemoverFactory()
        self.stop_words = set(stop_factory.get_stop_words())
        
        # Tambahan stopwords khusus
        additional_stopwords = {
            'yg', 'dgn', 'nya', 'kalo', 'kalau', 'udah', 'udh', 'dah', 
            'lg', 'lagi', 'banget', 'bgt', 'emang', 'memang', 'sih',
            'aja', 'doang', 'nih', 'nah', 'lah', 'deh', 'dong', 'kok',
            'ya', 'yah', 'wkwk', 'haha', 'hihi', 'huhu', 'hehe'
        }
        self.stop_words.update(additional_stopwords)
        
        # Hapus kata negasi dari stopwords agar tidak ikut dibuang
        # (penting untuk menjaga makna kalimat seperti "tidak bagus", "bukan salah", dll.)
        negation_words = {
            'tidak', 'bukan', 'belum', 'jangan', 'tak', 'tanpa',
            'kurang', 'jarang', 'hampir', 'nyaris'
        }
        self.stop_words -= negation_words
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )
        
        # SVM Model (Kernel RBF)
        self.model = SVC(kernel='rbf', C=1.0, gamma=1, probability=True)
        
        # Kamus normalisasi bahasa Indonesia
        self.normalization_dict = {
            'yg': 'yang', 'dgn': 'dengan', 'krn': 'karena', 'krna': 'karena',
            'tp': 'tapi', 'tpi': 'tapi', 'gk': 'tidak', 'ga': 'tidak',
            'gak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'g': 'tidak',
            'tdk': 'tidak', 'gitu': 'begitu', 'gt': 'begitu', 'gmn': 'bagaimana',
            'gimana': 'bagaimana', 'dmn': 'dimana',
            'kmn': 'kemana',
            'knp': 'kenapa', 'knapa': 'kenapa', 'org': 'orang', 'orng': 'orang',
            'tmn': 'teman', 'temen': 'teman', 'bgmn': 'bagaimana', 'bgt': 'banget',
            'banget': 'sangat', 'bener': 'benar', 'bnr': 'benar', 'bnyk': 'banyak',
            'bnyak': 'banyak', 'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah',
            'telah': 'sudah', 'blm': 'belum', 'blom': 'belum', 'msh': 'masih',
            'msih': 'masih', 'lg': 'lagi', 'lgi': 'lagi', 'skrg': 'sekarang',
            'skrang': 'sekarang', 'skg': 'sekarang', 'nanti': 'nanti',
            'ntar': 'nanti', 'tar': 'nanti', 'bsk': 'besok', 'besok': 'besok',
            'kmrn': 'kemarin', 'kmarin': 'kemarin', 'hrs': 'harus',
            'kudu': 'harus', 'mesti': 'harus', 'bs': 'bisa', 'bsa': 'bisa',
            'isa': 'bisa', 'biar': 'agar', 'spy': 'agar', 'supaya': 'agar',
            'kalo': 'kalau', 'klo': 'kalau', 'jd': 'jadi', 'jadi': 'menjadi',
            'jdnya': 'jadinya', 'jadinya': 'akhirnya', 'jg': 'juga', 'jga': 'juga',
            'jgn': 'jangan', 'jngn': 'jangan', 'jgn2': 'jangan-jangan',
            'aj': 'saja', 'aja': 'saja', 'doang': 'saja', 'aje': 'saja',
            'cm': 'cuma', 'cuma': 'hanya', 'cman': 'hanya', 'ckp': 'cukup',
            'cukup': 'cukup', 'krg': 'kurang', 'kurang': 'kurang', 'emg': 'memang',
            'emang': 'memang', 'mmg': 'memang', 'sbnrnya': 'sebenarnya',
            'sbenernya': 'sebenarnya', 'pdhl': 'padahal', 'pdahal': 'padahal',
            'wlpn': 'walaupun', 'walaupun': 'walaupun', 'meskipun': 'walaupun',
            'walau': 'walaupun', 'aplg': 'apalagi', 'apalagi': 'apalagi',
            'mgkn': 'mungkin', 'mungkin': 'mungkin', 'mgkin': 'mungkin',
            'kyknya': 'kayaknya', 'kyaknya': 'kayaknya', 'kayaknya': 'sepertinya',
            'kyk': 'seperti', 'kayak': 'seperti', 'ky': 'seperti', 'sprt': 'seperti',
            'kaya': 'seperti', 'sy': 'saya', 'gw': 'saya', 'gue': 'saya',
            'gua': 'saya', 'w': 'saya', 'aku': 'saya', 'ak': 'saya', 'km': 'kamu',
            'kmu': 'kamu', 'lu': 'kamu', 'lo': 'kamu', 'elu': 'kamu', 'elo': 'kamu',
            'u': 'kamu', 'dy': 'dia', 'dia': 'dia', 'mrk': 'mereka',
            'mreka': 'mereka', 'tololl': 'bodoh', 'tolol': 'bodoh',
            'qt': 'kita', 'qta': 'kita', 'seneng': 'senang', 'suka': 'suka',
            'sk': 'suka', 'kesel': 'kesal', 'binun': 'bingung', 'males': 'malas',
            'capek': 'capek', 'cape': 'capek', 'lelah': 'lelah', 'tired': 'lelah',
            'stress': 'stres', 'mantul': 'mantap', 'keren': 'keren', 'gokil': 'keren',
            'ajib': 'keren', 'top': 'bagus', 'the best': 'terbaik',
            'terbaik': 'terbaik', 'terburuk': 'terburuk', 'worst': 'terburuk',
            'best': 'terbaik', 'good': 'bagus', 'bad': 'buruk', 'nice': 'bagus',
            'awesome': 'keren', 'amazing': 'menakjubkan', 'terrible': 'buruk',
            'horrible': 'mengerikan', 'excellent': 'sangat bagus', 'perfect': 'sempurna',
            'ok': 'baik', 'oke': 'baik', 'okay': 'baik', 'fine': 'baik', 'standard': 'standar',
            'ajg':'anjing', 'anjg':'anjing', 'tw':'tau', 'kek':'seperti'
        }
        
    def text_cleaning(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def case_folding(self, text):
        return text.lower()
    
    def tokenizing(self, text):
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 1 and token.isalpha()]
        return tokens
    
    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def normalization(self, tokens):
        normalized_tokens = []
        for token in tokens:
            if token in self.normalization_dict:
                normalized_tokens.append(self.normalization_dict[token])
            else:
                normalized_tokens.append(token)
        return normalized_tokens
    
    def stemming(self, tokens):
        text = ' '.join(tokens)
        stemmed_text = self.stemmer.stem(text)
        return stemmed_text.split()
    
    def preprocess_text(self, text, show_steps=False):
        """
        Preprocessing dengan urutan:
        1. Cleaning
        2. Case Folding
        3. Tokenizing
        4. Normalization
        5. Stopwords Removal
        6. Stemming
        """
        steps = {}
        # Step 1: Cleaning
        cleaned = self.text_cleaning(text)
        if show_steps: steps['cleaned'] = cleaned
        # Step 2: Case Folding
        casefolded = self.case_folding(cleaned)
        if show_steps: steps['casefolded'] = casefolded
        # Step 3: Tokenizing
        tokens = self.tokenizing(casefolded)
        if show_steps: steps['tokenized'] = tokens
        # Step 4: Normalization
        normalized = self.normalization(tokens)
        if show_steps: steps['normalized'] = normalized
        # Step 5: Remove Stopwords
        no_stopwords = self.remove_stopwords(normalized)
        if show_steps: steps['no_stopwords'] = no_stopwords
        # Step 6: Stemming
        stemmed = self.stemming(no_stopwords)
        if show_steps: steps['stemmed'] = stemmed
        final_text = ' '.join(stemmed)
        if show_steps:
            steps['original'] = text
            steps['final'] = final_text
            return final_text, steps
        return final_text
    
    def load_and_preprocess_data(self, filepath):
        print(f"Loading dataset from {filepath}...")
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            print("⚠ UTF-8 gagal, mencoba encoding latin-1...")
            df = pd.read_csv(filepath, encoding='latin-1')
        
        print("Preprocessing texts...")
        df['processed_text'] = df['text'].apply(lambda x: self.preprocess_text(x))
        df = df[df['processed_text'].str.len() > 0]
        df['sentiment'] = df['sentiment'].astype(int)
        
        print("Preprocessing complete.")
        return df
    

    def print_confusion_matrix(self, y_test, y_pred, title="Confusion Matrix"):
        """
        Menampilkan confusion matrix dengan format yang jelas
        """
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{title}")
        print("="*60)
        print(f"\nDetail Metrik dari Confusion Matrix:")
        print(f" * True Positive (TP)  : {tp:<5} (Prediksi: Positif, Aktual: Positif)")
        print(f" * True Negative (TN)  : {tn:<5} (Prediksi: Negatif, Aktual: Negatif)")
        print(f" * False Positive (FP) : {fp:<5} (Prediksi: Positif, Aktual: Negatif) -> Error Tipe I")
        print(f" * False Negative (FN) : {fn:<5} (Prediksi: Negatif, Aktual: Positif) -> Error Tipe II")
        
        print("\nMatriks Konfusi (Visual):")
        print("                     Prediksi Negatif | Prediksi Positif")
        print("---------------------------------------------------------")
        print(f"Aktual Negatif (0) |     {tn:<10} |     {fp:<10}")
        print(f"Aktual Positif (1) |     {fn:<10} |     {tp:<10}")
        print("---------------------------------------------------------")
    
    def train_and_evaluate_model(self, df):
        """
        Training model dan evaluasi performa
        """
        print("\n" + "="*60)
        print("TRAINING MODEL DENGAN DATA ORIGINAL")
        print("="*60)
        
        X = df['processed_text']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train SVM
        print("Training SVM model...")
        self.model.fit(X_train_tfidf, y_train)
        print("✓ Training selesai!")
        
        # Evaluasi Model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        self.print_confusion_matrix(y_test, y_pred, "Confusion Matrix")
        
        return accuracy

    def predict_sentiment(self, text):
        processed_text = self.preprocess_text(text)
        if not processed_text.strip():
            return { 
                'sentiment': 'Tidak dapat menentukan', 
                'confidence': 0.0,
                'probability_negative': 0.5, 
                'probability_positive': 0.5 
            }
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        sentiment_label = "Positif" if prediction == 1 else "Negatif"
        confidence = max(probability)
        return {
            'sentiment': sentiment_label, 
            'confidence': confidence,
            'probability_negative': probability[0], 
            'probability_positive': probability[1]
        }

    def save_model(self, filepath='sentiment_model.pkl'):
        model_data = {
            'model': self.model, 
            'vectorizer': self.vectorizer,
            'stemmer': self.stemmer, 
            'stop_words': self.stop_words,
            'normalization_dict': self.normalization_dict
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='sentiment_model.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.stemmer = model_data['stemmer']
        self.stop_words = model_data['stop_words']
        self.normalization_dict = model_data['normalization_dict']
        print(f"✓ Model loaded from {filepath}")

def main():
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS - MULTI-RATIO PERFORMANCE TEST")
    print("="*60)
    
    analyzer = SentimentAnalyzer()

    # 1. Load Data
    df = analyzer.load_and_preprocess_data('data_mbg_labelled.csv')
    
    # Save processed data for dashboard
    print("\nSaving processed data to mbg_processed.csv...")
    df.to_csv('mbg_processed.csv', index=False, encoding='utf-8')
    print("✓ Processed data saved successfully!")
    
    # ── GRID SEARCH: Cari C & gamma terbaik (dilakukan sekali sebelum loop rasio) ──
    print("\n" + "="*60)
    print(" GRID SEARCH - PENCARIAN C & GAMMA TERBAIK")
    print("="*60)

    X_all = df['processed_text']
    y_all = df['sentiment']

    # Split sementara 80:20 khusus untuk grid search
    X_gs_train, X_gs_test, y_gs_train, y_gs_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # TF-IDF untuk grid search
    from sklearn.pipeline import Pipeline
    gs_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                    lowercase=True, strip_accents='unicode')
    X_gs_tfidf = gs_vectorizer.fit_transform(X_gs_train)

    param_grid = {
        'C':     [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    print(f"Parameter yang diuji:")
    print(f"   C     : {param_grid['C']}")
    print(f"   gamma : {param_grid['gamma']}")
    print(f"   CV    : 3-fold (untuk efisiensi)")
    print(f"   Total kombinasi: {len(param_grid['C']) * len(param_grid['gamma'])} kombinasi × 3 fold")
    print("\nProses grid search sedang berjalan, harap tunggu...")

    grid_search = GridSearchCV(
        SVC(kernel='rbf', probability=True),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_gs_tfidf, y_gs_train)

    best_C     = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    best_cv_score = grid_search.best_score_

    print(f"\n✓ Grid Search selesai!")
    print(f"{'─'*45}")
    print(f" Hasil Grid Search:")
    print(f"   Best C        : {best_C}")
    print(f"   Best gamma    : {best_gamma}")
    print(f"   Best CV Score : {best_cv_score*100:.2f}% (rata-rata 3-fold)")
    print(f"{'─'*45}")

    # Tampilkan tabel semua kombinasi
    print(f"\n Ringkasan seluruh kombinasi:")
    print(f" {'C':<8} | {'gamma':<8} | {'CV Accuracy':>12}")
    print(f" {'-'*8}-+-{'-'*8}-+-{'-'*12}")
    gs_results = grid_search.cv_results_
    for c_val, g_val, score in zip(
            gs_results['param_C'], gs_results['param_gamma'], gs_results['mean_test_score']):
        marker = " ◄ TERBAIK" if (c_val == best_C and g_val == best_gamma) else ""
        print(f" {str(c_val):<8} | {str(g_val):<8} | {score*100:>11.2f}%{marker}")
    print(f"{'─'*45}")

    # Terapkan parameter terbaik ke model analyzer
    analyzer.model = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True)
    analyzer.vectorizer = gs_vectorizer
    print(f"\n✓ Model SVM diperbarui dengan C={best_C}, gamma={best_gamma}")

    # List rasio yang akan diuji (test_size adalah kebalikannya)
    # 0.1 = 90:10, 0.2 = 80:20, 0.3 = 70:30
    test_ratios = [0.1, 0.2, 0.3]
    results = []
    best_accuracy = 0
    best_ratio = None
    best_test_size = None

    for test_size in test_ratios:
        ratio_name = f"{int((1-test_size)*100)}:{int(test_size*100)}"
        print("\n\n" + "#"*70)
        print(f" PENGUJIAN RASIO DATA {ratio_name}")
        print("#"*70)
        
        X = df['processed_text']
        y = df['sentiment']
        
        # Split data dengan rasio saat ini
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Jumlah Data Training: {len(X_train)}")
        print(f"Jumlah Data Testing : {len(X_test)}")
        
        # TF-IDF
        X_train_tfidf = analyzer.vectorizer.fit_transform(X_train)
        X_test_tfidf = analyzer.vectorizer.transform(X_test)
        
        # Training
        analyzer.model.fit(X_train_tfidf, y_train)
        
        # Prediksi
        y_pred = analyzer.model.predict(X_test_tfidf)
        
        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Tampilkan Confusion Matrix Visual
        print(f"\nConfusion Matrix ({ratio_name}):")
        print(f"{'':<20} | {'Pred Negatif':<14} | {'Pred Positif':<14}")
        print("-" * 55)
        print(f"{'Aktual Negatif (0)':<20} | {tn:<14} | {fp:<14}")
        print(f"{'Aktual Positif (1)':<20} | {fn:<14} | {tp:<14}")
        print("-" * 55)
        print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"  Total data uji = TP+TN+FP+FN = {tp}+{tn}+{fp}+{fn} = {tp+tn+fp+fn}")

        # ── PERHITUNGAN RUNTUT SETIAP METRIK ──
        total = tp + tn + fp + fn

        # --- Kelas NEGATIF ---
        prec_neg  = report['Negatif']['precision']
        rec_neg   = report['Negatif']['recall']
        f1_neg    = report['Negatif']['f1-score']
        sup_neg   = int(report['Negatif']['support'])

        # --- Kelas POSITIF ---
        prec_pos  = report['Positif']['precision']
        rec_pos   = report['Positif']['recall']
        f1_pos    = report['Positif']['f1-score']
        sup_pos   = int(report['Positif']['support'])

        print(f"\n{'─'*60}")
        print(f" PERHITUNGAN METRIK EVALUASI  (Rasio {ratio_name})")
        print(f"{'─'*60}")

        print(f"\n▶ PRECISION")
        print(f"   Rumus  : TP / (TP + FP)  [per kelas]")
        print(f"   Negatif: TN / (TN + FN) = {tn} / ({tn}+{fn}) = {tn}/{tn+fn} = {prec_neg*100:.2f}%")
        print(f"   Positif: TP / (TP + FP) = {tp} / ({tp}+{fp}) = {tp}/{tp+fp} = {prec_pos*100:.2f}%")
        prec_w = report['weighted avg']['precision']
        print(f"   Weighted Avg = ({prec_neg:.6f}×{sup_neg} + {prec_pos:.6f}×{sup_pos}) / {total}")
        print(f"                = {prec_neg*sup_neg:.4f} + {prec_pos*sup_pos:.4f} / {total}")
        print(f"                = {prec_w*100:.2f}%")

        print(f"\n▶ RECALL")
        print(f"   Rumus  : TP / (TP + FN)  [per kelas]")
        print(f"   Negatif: TN / (TN + FP) = {tn} / ({tn}+{fp}) = {tn}/{tn+fp} = {rec_neg*100:.2f}%")
        print(f"   Positif: TP / (TP + FN) = {tp} / ({tp}+{fn}) = {tp}/{tp+fn} = {rec_pos*100:.2f}%")
        rec_w = report['weighted avg']['recall']
        print(f"   Weighted Avg = ({rec_neg:.6f}×{sup_neg} + {rec_pos:.6f}×{sup_pos}) / {total}")
        print(f"                = {rec_neg*sup_neg:.4f} + {rec_pos*sup_pos:.4f} / {total}")
        print(f"                = {rec_w*100:.2f}%")

        print(f"\n▶ F1-SCORE")
        print(f"   Rumus  : 2 × (Precision × Recall) / (Precision + Recall)  [per kelas]")
        print(f"   Negatif: 2×({prec_neg:.6f}×{rec_neg:.6f}) / ({prec_neg:.6f}+{rec_neg:.6f})")
        print(f"          = 2×{prec_neg*rec_neg:.6f} / {prec_neg+rec_neg:.6f} = {f1_neg*100:.2f}%")
        print(f"   Positif: 2×({prec_pos:.6f}×{rec_pos:.6f}) / ({prec_pos:.6f}+{rec_pos:.6f})")
        print(f"          = 2×{prec_pos*rec_pos:.6f} / {prec_pos+rec_pos:.6f} = {f1_pos*100:.2f}%")
        f1_w = report['weighted avg']['f1-score']
        print(f"   Weighted Avg = ({f1_neg:.6f}×{sup_neg} + {f1_pos:.6f}×{sup_pos}) / {total}")
        print(f"                = {f1_neg*sup_neg:.4f} + {f1_pos*sup_pos:.4f} / {total}")
        print(f"                = {f1_w*100:.2f}%")

        print(f"\n▶ ACCURACY")
        print(f"   Rumus  : (TP + TN) / (TP + TN + FP + FN)")
        print(f"          = ({tp} + {tn}) / ({tp}+{tn}+{fp}+{fn})")
        print(f"          = {tp+tn} / {total}")
        print(f"          = {acc*100:.2f}%")

        print(f"\n{'─'*60}")
        print(f" RINGKASAN HASIL AKHIR  (Rasio {ratio_name})")
        print(f"{'─'*60}")
        print(f" {'Kelas':<12} | {'Precision':>12} | {'Recall':>12} | {'F1-Score':>12} | {'Support':>8}")
        print(f" {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
        print(f" {'Negatif':<12} | {prec_neg*100:>11.2f}% | {rec_neg*100:>11.2f}% | {f1_neg*100:>11.2f}% | {sup_neg:>8}")
        print(f" {'Positif':<12} | {prec_pos*100:>11.2f}% | {rec_pos*100:>11.2f}% | {f1_pos*100:>11.2f}% | {sup_pos:>8}")
        print(f" {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
        print(f" {'Weighted Avg':<12} | {prec_w*100:>11.2f}% | {rec_w*100:>11.2f}% | {f1_w*100:>11.2f}% | {total:>8}")
        print(f"\n Accuracy: {acc*100:.2f}%")
        print(f"{'─'*60}")

        # Simpan ringkasan untuk tabel akhir
        results.append({
            'Rasio': ratio_name,
            'Accuracy': acc,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score']
        })
        
        # Cek apakah ini rasio terbaik
        if acc > best_accuracy:
            best_accuracy = acc
            best_ratio = ratio_name
            best_test_size = test_size

    # 2. Ringkasan Akhir untuk Tabel Skripsi
    print("\n\n" + "="*70)
    print(" RINGKASAN PERFORMA SEMUA RASIO (UNTUK TABEL SKRIPSI)")
    print("="*70)
    print(f" {'Rasio':<8} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
    print(f" {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for r in results:
        print(f" {r['Rasio']:<8} | {r['Accuracy']*100:>9.2f}% | {r['Precision']*100:>9.2f}% | {r['Recall']*100:>9.2f}% | {r['F1-Score']*100:>9.2f}%")
    print("="*70)

    # 3. Tampilkan Rasio Terbaik
    print("\n" + "="*70)
    print(" RASIO TERBAIK BERDASARKAN ACCURACY")
    print("="*70)
    print(f" Rasio Terbaik : {best_ratio}")
    print(f" Accuracy      : {best_accuracy*100:.2f}%")
    print("="*70)

    # 4. Train ulang model dengan rasio terbaik dan simpan
    print(f"\n🔄 Training ulang model dengan rasio terbaik ({best_ratio})...")
    X = df['processed_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=best_test_size, random_state=42, stratify=y
    )
    # TF-IDF
    X_train_tfidf = analyzer.vectorizer.fit_transform(X_train)
    X_test_tfidf = analyzer.vectorizer.transform(X_test)
    # Training
    analyzer.model.fit(X_train_tfidf, y_train)

    # ── NILAI ALPHA (Lagrange Multiplier) SVM ──
    # dual_coef_ berisi α_i × y_i untuk setiap support vector
    dual_coefs = analyzer.model.dual_coef_          # shape: (n_classes-1, n_support_vectors)
    alpha_values = np.abs(np.asarray(dual_coefs.todense())).flatten()     # ambil nilai absolut α_i
    n_sv = analyzer.model.support_vectors_.shape[0] # jumlah total support vector

    avg_alpha = float(np.mean(alpha_values))
    min_alpha = float(np.min(alpha_values))
    max_alpha = float(np.max(alpha_values))

    print(f"\n{'─'*55}")
    print(f" NILAI ALPHA (Lagrange Multiplier) MODEL SVM")
    print(f"{'─'*55}")
    print(f"   Keterangan   : α_i diperoleh dari |dual_coef_| model")
    print(f"   Jumlah Support Vector : {n_sv}")
    print(f"   Rata-rata α  : {avg_alpha:.6f}")
    print(f"   α Minimum    : {min_alpha:.6f}")
    print(f"   α Maksimum   : {max_alpha:.6f}")
    print(f"{'─'*55}")
    print(f"   ★ Gunakan nilai Rata-rata α = {avg_alpha:.6f}")
    print(f"     sebagai given value untuk hitungan manual SVM di skripsi.")
    print(f"{'─'*55}")

    # Calculate performance metrics for the best model
    y_pred = analyzer.model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'], output_dict=True)

    tn_b, fp_b, fn_b, tp_b = cm.ravel()
    total_b = int(tp_b + tn_b + fp_b + fn_b)

    prec_neg_b = report['Negatif']['precision']
    rec_neg_b  = report['Negatif']['recall']
    f1_neg_b   = report['Negatif']['f1-score']
    sup_neg_b  = int(report['Negatif']['support'])

    prec_pos_b = report['Positif']['precision']
    rec_pos_b  = report['Positif']['recall']
    f1_pos_b   = report['Positif']['f1-score']
    sup_pos_b  = int(report['Positif']['support'])

    prec_w_b = report['weighted avg']['precision']
    rec_w_b  = report['weighted avg']['recall']
    f1_w_b   = report['weighted avg']['f1-score']

    print(f"\n{'='*65}")
    print(f" EVALUASI MODEL TERBAIK  (Rasio {best_ratio})")
    print(f"{'='*65}")

    print(f"\n Confusion Matrix:")
    print(f"   {'':<22} | {'Pred Negatif':<14} | {'Pred Positif':<14}")
    print(f"   {'-'*22}-+-{'-'*14}-+-{'-'*14}")
    print(f"   {'Aktual Negatif (0)':<22} | {tn_b:<14} | {fp_b:<14}")
    print(f"   {'Aktual Positif (1)':<22} | {fn_b:<14} | {tp_b:<14}")
    print(f"   {'-'*22}-+-{'-'*14}-+-{'-'*14}")
    print(f"   TP={tp_b}, TN={tn_b}, FP={fp_b}, FN={fn_b}  →  Total uji = {total_b}")

    print(f"\n{'─'*65}")
    print(f" PERHITUNGAN RUNTUT SETIAP METRIK")
    print(f"{'─'*65}")

    print(f"\n▶ PRECISION  (seberapa tepat prediksi positif/negatif model)")
    print(f"   Rumus per kelas: jumlah prediksi benar kelas X / semua prediksi kelas X")
    print(f"   Negatif : TN / (TN+FN) = {tn_b} / ({tn_b}+{fn_b}) = {tn_b}/{tn_b+fn_b} = {prec_neg_b*100:.2f}%")
    print(f"   Positif : TP / (TP+FP) = {tp_b} / ({tp_b}+{fp_b}) = {tp_b}/{tp_b+fp_b} = {prec_pos_b*100:.2f}%")
    print(f"   Weighted Avg:")
    print(f"     = (Precision_Neg × Support_Neg + Precision_Pos × Support_Pos) / Total")
    print(f"     = ({prec_neg_b:.6f} × {sup_neg_b} + {prec_pos_b:.6f} × {sup_pos_b}) / {total_b}")
    print(f"     = ({prec_neg_b*sup_neg_b:.4f} + {prec_pos_b*sup_pos_b:.4f}) / {total_b}")
    print(f"     = {prec_neg_b*sup_neg_b + prec_pos_b*sup_pos_b:.4f} / {total_b}")
    print(f"     = {prec_w_b*100:.2f}%")

    print(f"\n▶ RECALL  (seberapa banyak data aktual yang berhasil ditemukan model)")
    print(f"   Rumus per kelas: jumlah prediksi benar kelas X / semua data aktual kelas X")
    print(f"   Negatif : TN / (TN+FP) = {tn_b} / ({tn_b}+{fp_b}) = {tn_b}/{tn_b+fp_b} = {rec_neg_b*100:.2f}%")
    print(f"   Positif : TP / (TP+FN) = {tp_b} / ({tp_b}+{fn_b}) = {tp_b}/{tp_b+fn_b} = {rec_pos_b*100:.2f}%")
    print(f"   Weighted Avg:")
    print(f"     = (Recall_Neg × Support_Neg + Recall_Pos × Support_Pos) / Total")
    print(f"     = ({rec_neg_b:.6f} × {sup_neg_b} + {rec_pos_b:.6f} × {sup_pos_b}) / {total_b}")
    print(f"     = ({rec_neg_b*sup_neg_b:.4f} + {rec_pos_b*sup_pos_b:.4f}) / {total_b}")
    print(f"     = {rec_neg_b*sup_neg_b + rec_pos_b*sup_pos_b:.4f} / {total_b}")
    print(f"     = {rec_w_b*100:.2f}%")

    print(f"\n▶ F1-SCORE  (harmonic mean antara Precision dan Recall)")
    print(f"   Rumus per kelas: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   Negatif:")
    print(f"     = 2 × ({prec_neg_b:.6f} × {rec_neg_b:.6f}) / ({prec_neg_b:.6f} + {rec_neg_b:.6f})")
    print(f"     = 2 × {prec_neg_b*rec_neg_b:.6f} / {prec_neg_b+rec_neg_b:.6f}")
    print(f"     = {2*prec_neg_b*rec_neg_b:.6f} / {prec_neg_b+rec_neg_b:.6f}")
    print(f"     = {f1_neg_b*100:.2f}%")
    print(f"   Positif:")
    print(f"     = 2 × ({prec_pos_b:.6f} × {rec_pos_b:.6f}) / ({prec_pos_b:.6f} + {rec_pos_b:.6f})")
    print(f"     = 2 × {prec_pos_b*rec_pos_b:.6f} / {prec_pos_b+rec_pos_b:.6f}")
    print(f"     = {2*prec_pos_b*rec_pos_b:.6f} / {prec_pos_b+rec_pos_b:.6f}")
    print(f"     = {f1_pos_b*100:.2f}%")
    print(f"   Weighted Avg:")
    print(f"     = (F1_Neg × Support_Neg + F1_Pos × Support_Pos) / Total")
    print(f"     = ({f1_neg_b:.6f} × {sup_neg_b} + {f1_pos_b:.6f} × {sup_pos_b}) / {total_b}")
    print(f"     = ({f1_neg_b*sup_neg_b:.4f} + {f1_pos_b*sup_pos_b:.4f}) / {total_b}")
    print(f"     = {f1_neg_b*sup_neg_b + f1_pos_b*sup_pos_b:.4f} / {total_b}")
    print(f"     = {f1_w_b*100:.2f}%")

    print(f"\n▶ ACCURACY  (proporsi prediksi yang benar dari seluruh data uji)")
    print(f"   Rumus: (TP + TN) / (TP + TN + FP + FN)")
    print(f"        = ({tp_b} + {tn_b}) / ({tp_b} + {tn_b} + {fp_b} + {fn_b})")
    print(f"        = {tp_b+tn_b} / {total_b}")
    print(f"        = {accuracy*100:.2f}%")

    print(f"\n{'─'*65}")
    print(f" TABEL HASIL EVALUASI AKHIR  (Rasio Terbaik: {best_ratio})")
    print(f"{'─'*65}")
    print(f" {'Kelas':<14} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>8}")
    print(f" {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f" {'Negatif':<14} | {prec_neg_b*100:>9.2f}% | {rec_neg_b*100:>9.2f}% | {f1_neg_b*100:>9.2f}% | {sup_neg_b:>8}")
    print(f" {'Positif':<14} | {prec_pos_b*100:>9.2f}% | {rec_pos_b*100:>9.2f}% | {f1_pos_b*100:>9.2f}% | {sup_pos_b:>8}")
    print(f" {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f" {'Weighted Avg':<14} | {prec_w_b*100:>9.2f}% | {rec_w_b*100:>9.2f}% | {f1_w_b*100:>9.2f}% | {total_b:>8}")
    print(f"\n {'Accuracy':>47} : {accuracy*100:.2f}%")
    print(f"{'='*65}")

    # Save performance metrics to JSON
    import json
    metrics = {
        'best_ratio': best_ratio,
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'true_negative': int(tn_b),
            'false_positive': int(fp_b),
            'false_negative': int(fn_b),
            'true_positive': int(tp_b)
        },
        'classification_report': {
            'negative': {
                'precision': float(prec_neg_b),
                'recall': float(rec_neg_b),
                'f1-score': float(f1_neg_b)
            },
            'positive': {
                'precision': float(prec_pos_b),
                'recall': float(rec_pos_b),
                'f1-score': float(f1_pos_b)
            },
            'weighted_avg': {
                'precision': float(prec_w_b),
                'recall': float(rec_w_b),
                'f1-score': float(f1_w_b)
            }
        }
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("✓ Model metrics saved to model_metrics.json")
    
    # Simpan model dengan rasio terbaik
    analyzer.save_model('sentiment_model.pkl')
    print(f"✓ Model dengan rasio {best_ratio} berhasil disimpan!")

if __name__ == "__main__":
    main()