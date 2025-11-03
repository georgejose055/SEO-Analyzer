import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
import os
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings("ignore")

# Try imports for optional libs (graceful fallback)
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available; using simple splitting for sentences/words.")
    def sent_tokenize(text):
        return text.split('. ') if text else []
    def word_tokenize(text):
        return text.split() if text else []

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    st.warning("SentenceTransformers not available; similarity as keyword matches.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("sklearn not available; quality as 'Medium' fallback.")
    def RandomForestClassifier(): pass
    def LabelEncoder(): pass

# Predefined SEO references
REFERENCE_SNIPPETS = [
    "SEO optimization improves search engine rankings through keyword density and content quality.",
    "High-quality content with good readability scores enhances user engagement and reduces bounce rates.",
    "Thin content with low word count often leads to poor SEO performance and penalties.",
    "Flesch Reading Ease score between 60-70 indicates easy-to-read content ideal for web audiences.",
    "Meta tags, alt text, header structure, and internal linking boost SEO signals for better visibility.",
    "Long-form content over 1000 words typically ranks higher if structured with subheadings and lists.",
    "Mobile-friendly design and fast load times are critical SEO factors in 2025 algorithms.",
    "Backlinks from authoritative sites and social signals contribute to domain authority."
]

# Simple NLTK download fallback
def ensure_nltk_data():
    if NLTK_AVAILABLE:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    return True

# Generate realistic sample CSV (23 High, 23 Medium, 23 Low; based on SEO thresholds)
@st.cache_data
def generate_sample_csv():
    np.random.seed(42)
    n_high = 23; n_medium = 23; n_low = 23
    data = []
    # High: Good metrics
    data.extend([{
        'word_count': np.random.randint(800, 3000, 1)[0],
        'flesch_reading_ease': np.random.uniform(60, 90, 1)[0],
        'sentence_count': np.random.randint(50, 150, 1)[0],
        'avg_sentence_length': np.random.uniform(15, 25, 1)[0],
        'thin_content': False,
        'quality_label': 'High'
    } for _ in range(n_high)])
    # Medium: Average
    data.extend([{
        'word_count': np.random.randint(400, 800, 1)[0],
        'flesch_reading_ease': np.random.uniform(30, 60, 1)[0],
        'sentence_count': np.random.randint(20, 50, 1)[0],
        'avg_sentence_length': np.random.uniform(20, 35, 1)[0],
        'thin_content': np.random.choice([True, False], p=[0.4, 0.6]),
        'quality_label': 'Medium'
    } for _ in range(n_medium)])
    # Low: Poor/thin
    data.extend([{
        'word_count': np.random.randint(100, 400, 1)[0],
        'flesch_reading_ease': np.random.uniform(10, 30, 1)[0],
        'sentence_count': np.random.randint(5, 20, 1)[0],
        'avg_sentence_length': np.random.uniform(25, 50, 1)[0],
        'thin_content': True,
        'quality_label': 'Low'
    } for _ in range(n_low)])
    df = pd.DataFrame(data)
    csv_path = 'processed_data.csv'
    df.to_csv(csv_path, index=False)
    st.info("Generated sample processed_data.csv (69 rows, balanced). Replace with real data.")
    return df, csv_path

# Load or train models (no SMOTE; simple fit)
@st.cache_resource
def load_or_train_models():
    model_file = 'rf_model.pkl'
    encoder_file = 'label_encoder.pkl'
    csv_file = 'processed_data.csv'
    
    # Try load
    loaded = False
    if os.path.exists(model_file) and os.path.exists(encoder_file):
        try:
            with open(model_file, 'rb') as f:
                rf_model = pickle.load(f)
            with open(encoder_file, 'rb') as f:
                label_encoder = pickle.load(f)
            loaded = True
            st.success("Models loaded from files.")
        except Exception as e:
            st.warning(f"Load error: {e}. Training dynamically.")
    
    if not loaded or not os.path.exists(csv_file):
        df, csv_file = generate_sample_csv()
    else:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            df, csv_file = generate_sample_csv()
    
    if ML_AVAILABLE:
        feature_cols = ['word_count', 'flesch_reading_ease', 'sentence_count', 'avg_sentence_length', 'thin_content']
        # Ensure features
        for col in feature_cols:
            if col not in df.columns:
                if col == 'thin_content':
                    df[col] = df['word_count'] < 300
                else:
                    df[col] = 50  # Default
        X = df[feature_cols].fillna(0).values
        y = df['quality_label']
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Smaller for speed
        rf_model.fit(X, y_encoded)
        
        # Save
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(rf_model, f)
            with open(encoder_file, 'wb') as f:
                pickle.dump(label_encoder, f)
        except Exception:
            pass  # Ignore save errors
        
        st.info("Models trained dynamically.")
        return rf_model, label_encoder, csv_file
    else:
        st.warning("ML libs missing; default 'Medium' label.")
        return None, None, csv_file

# Features computation
def compute_features(text):
    ensure_nltk_data()
    sentences = sent_tokenize(text)
    words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)
    try:
        flesch_score = flesch_reading_ease(text)
    except:
        flesch_score = 50  # Fallback
    thin_content = word_count < 300
    
    return {
        'word_count': word_count,
        'flesch_reading_ease': round(flesch_score, 1),
        'sentence_count': sentence_count,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'thin_content': thin_content
    }

# Fetch content
def fetch_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)[:5000]
        return ' '.join(text.split())
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        return "Sample text for demo."

# Similarity (SBERT or keyword fallback)
def compute_similarity(text, _=None):
    if not SBERT_AVAILABLE:
        # Keyword fallback
        keywords = ['seo', 'content', 'ranking', 'readability', 'keywords', 'backlinks']
        matches = [kw for kw in keywords if kw in text.lower()]
        return [f"Matches '{kw}' in SEO context." for kw in matches[:3]]
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = sent_tokenize(text)[:5]
        if sentences:
            doc_emb = model.encode(sentences)
            ref_emb = model.encode(REFERENCE_SNIPPETS)
            sims = cosine_similarity(doc_emb.mean(axis=0).reshape(1,-1), ref_emb)[0]
            top_idx = np.argsort(sims)[::-1][:3]
            return [REFERENCE_SNIPPETS[i] for i in top_idx if sims[i] > 0.2]
    except Exception as e:
        st.warning(f"Similarity error: {e}; using keywords.")
        return compute_similarity(text)  # Fallback to keyword
    return []

# Main
def main():
    st.title("🔍 SEO Analyzer")
    st.write("Analyze URL for content quality and SEO insights.")
    
    url = st.text_input("URL", "https://example.com")
    
    if st.button("Analyze"):
        text = fetch_content(url)
        features = compute_features(text)
        similar_to = compute_similarity(text)
        
        thin_val = 1 if features['thin_content'] else 0
        feature_vector = np.array([[features['word_count'], features['flesch_reading_ease'],
                                    features['sentence_count'], features['avg_sentence_length'], thin_val]])
        
        rf_model, label_encoder, _ = load_or_train_models()
        if ML_AVAILABLE and rf_model and label_encoder:
            pred = rf_model.predict(feature_vector)[0]
            quality_label = label_encoder.inverse_transform([pred])[0]
        else:
            quality_label = "Medium"  # Fallback
        
        result = {
            "quality_label": quality_label,
            "word_count": features['word_count'],
            "flesch_reading_ease": features['flesch_reading_ease'],
            "sentence_count": features['sentence_count'],
            "avg_sentence_length": features['avg_sentence_length'],
            "thin_content": features['thin_content'],
            "similar_to": similar_to[:3]
        }
        
        st.json(result)
        st.success("Done!")

if __name__ == "__main__":
    main()
