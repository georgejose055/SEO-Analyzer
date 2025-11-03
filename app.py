import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
import os
from textstat import flesch_reading_ease
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from imblearn.over_sampling import SMOTE  # For balancing; optional

# Download NLTK data (persistent cache)
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    return True

# Predefined SEO reference snippets (expanded for better similarity)
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

# Generate sample CSV if missing (69 rows: mix High/Medium/Low quality)
def generate_sample_csv():
    np.random.seed(42)
    n_samples = 69
    data = {
        'word_count': np.random.randint(100, 3000, n_samples),
        'flesch_reading_ease': np.random.uniform(10, 90, n_samples),
        'sentence_count': np.random.randint(5, 200, n_samples),
        'avg_sentence_length': np.random.uniform(10, 40, n_samples),
        'thin_content': np.random.choice([False, True], n_samples, p=[0.7, 0.3])
    }
    # Simulate labels based on features (High: good scores; Medium: avg; Low: poor/thin)
    df = pd.DataFrame(data)
    df['quality_label'] = 'Low'
    df.loc[(df['word_count'] > 800) & (df['flesch_reading_ease'] > 50) & ~df['thin_content'], 'quality_label'] = 'High'
    df.loc[(df['word_count'] > 400) & (df['flesch_reading_ease'] > 30) | df['thin_content'], 'quality_label'] = 'Medium'
    df.to_csv('processed_data.csv', index=False)
    st.info("Generated sample processed_data.csv (69 rows). Train your own for accuracy.")
    return df

# Load or train models dynamically (robust: generates CSV if missing)
@st.cache_resource
def load_or_train_models():
    model_file = 'rf_model.pkl'
    encoder_file = 'label_encoder.pkl'
    csv_file = 'processed_data.csv'
    
    # Try load from files
    if os.path.exists(model_file) and os.path.exists(encoder_file) and os.path.exists(csv_file):
        try:
            with open(model_file, 'rb') as f:
                rf_model = pickle.load(f)
            with open(encoder_file, 'rb') as f:
                label_encoder = pickle.load(f)
            st.success("Models loaded from .pkl files.")
            return rf_model, label_encoder, True, csv_file  # Loaded
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            st.warning(f"Model/CSV load failed ({e}); using dynamic training.")
    
    # Dynamic: Load or generate CSV
    if not os.path.exists(csv_file):
        df = generate_sample_csv()
    else:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            st.warning(f"CSV read failed ({e}); generating sample.")
            df = generate_sample_csv()
    
    if df.empty or 'quality_label' not in df.columns:
        st.error("CSV invalid/missing 'quality_label'. Generated sample.")
        df = generate_sample_csv()
    
    # Features (match your data; add more if CSV has extras)
    feature_cols = ['word_count', 'flesch_reading_ease', 'sentence_count', 'avg_sentence_length', 'thin_content']
    # Ensure columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col == 'thin_content':
                df[col] = df['word_count'] < 300
            else:
                df[col] = np.random.uniform(0, 100, len(df))  # Fallback; replace with real
    X = df[feature_cols].fillna(0).values  # NumPy array
    y = df['quality_label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Balance classes (SMOTE for small dataset)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y_encoded)
    
    # Train RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_balanced, y_balanced)
    
    # Save models (for future; ignores LFS for now)
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(rf_model, f)
        with open(encoder_file, 'wb') as f:
            pickle.dump(label_encoder, f)
        st.success("Models trained and saved. (Fix LFS for persistence.)")
    except Exception as e:
        st.warning(f"Save failed ({e}); models in memory.")
    
    return rf_model, label_encoder, False, csv_file  # Trained

# Compute SEO features
def compute_features(text):
    download_nltk_data()
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    word_count = len([w for w in words if w.isalpha()])  # Alpha words only
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)
    flesch_score = flesch_reading_ease(text)
    thin_content = word_count < 300
    
    return {
        'word_count': word_count,
        'flesch_reading_ease': flesch_score,
        'sentence_count': sentence_count,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'thin_content': thin_content
    }

# Fetch/clean URL content (improved: handle redirects, limit)
def fetch_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove unwanted
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Limit and clean
        text = ' '.join(text.split())[:8000]  # Chars limit
        return text
    except requests.RequestException as e:
        st.error(f"Fetch error for {url}: {e}")
        return ""
    except Exception as e:
        st.error(f"Parse error: {e}")
        return ""

# SBERT similarity (cached model)
@st.cache_resource
def load_sbert():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text, model):
    if len(text) < 50:
        return []
    sentences = sent_tokenize(text)[:10]  # Top 10 sentences
    if not sentences:
        return []
    doc_emb = model.encode(sentences)
    ref_emb = model.encode(REFERENCE_SNIPPETS)
    sims = cosine_similarity(doc_emb.mean(axis=0).reshape(1, -1), ref_emb)[0]
    top_indices = np.argsort(sims)[::-1][:3]
    return [REFERENCE_SNIPPETS[i] for i in top_indices if sims[i] > 0.3]  # Threshold for relevance

# Main app
def main():
    st.set_page_config(page_title="SEO Analyzer", page_icon="🔍")
    st.title("🔍 SEO Content Analyzer")
    st.write("Paste a URL to evaluate content quality, readability, and SEO similarity.")
    
    url = st.text_input("Enter URL", value="https://example.com", help="Must start with http(s)://")
    
    if st.button("Analyze Content", type="primary"):
        if not url or not any(url.startswith(prefix) for prefix in ['http://', 'https://']):
            st.error("Invalid URL. Use http:// or https://.")
            return
        
        with st.spinner("Analyzing... (Fetching, processing, ML inference)"):
            text = fetch_content(url)
            if not text:
                st.stop()
            
            features = compute_features(text)
            sbert_model = load_sbert()
            similar_to = compute_similarity(text, sbert_model)
            
            # Feature vector
            thin_val = 1 if features['thin_content'] else 0
            feature_vector = np.array([[features['word_count'], features['flesch_reading_ease'],
                                        features['sentence_count'], features['avg_sentence_length'], thin_val]])
            
            # Models
            rf_model, label_encoder, trained_dynamic, csv_path = load_or_train_models()
            if rf_model is None or label_encoder is None:
                st.error("Model training failed. Check logs.")
                return
            
            # Predict
            pred_encoded = rf_model.predict(feature_vector)[0]
            quality_label = label_encoder.inverse_transform([pred_encoded])[0]
            
            # Results
            result = {
                "quality_label": quality_label,
                "word_count": features['word_count'],
                "flesch_reading_ease": round(features['flesch_reading_ease'], 1),
                "sentence_count": features['sentence_count'],
                "avg_sentence_length": features['avg_sentence_length'],
                "thin_content": features['thin_content'],
                "similar_to": similar_to
            }
            
            st.subheader("Results")
            st.json(result)
            
            # Insights
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quality", quality_label)
                st.metric("Readability (Flesch)", f"{result['flesch_reading_ease']} / 100")
            with col2:
                st.metric("Word Count", result['word_count'])
                st.metric("Thin Content?", "Yes" if result['thin_content'] else "No")
            
            if similar_to:
                st.subheader("Similar SEO Topics")
                for sim in similar_to:
                    st.write(f"• {sim}")
            else:
                st.warning("No strong SEO similarities found; consider adding keywords.")
            
            if trained_dynamic:
                st.info("Used dynamic training (sample data). Upload real processed_data.csv for better accuracy.")
        
        st.success("Analysis done! Try another URL.")

if __name__ == "__main__":
    main()
