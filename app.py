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
from imblearn.over_sampling import SMOTE  # If imbalanced; optional

# Download NLTK data (cache with st.cache_data)
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    return True

# Predefined SEO reference sentences for similarity (expand as needed)
REFERENCE_SNIPPETS = [
    "SEO optimization improves search engine rankings through keyword density and content quality.",
    "High-quality content with good readability scores enhances user engagement and reduces bounce rates.",
    "Thin content with low word count often leads to poor SEO performance and penalties.",
    "Flesch Reading Ease score between 60-70 indicates easy-to-read content ideal for web.",
    "Meta tags, alt text, and internal linking boost SEO signals for better visibility."
]

# Load or train models dynamically
@st.cache_resource
def load_or_train_models():
    model_file = 'rf_model.pkl'
    encoder_file = 'label_encoder.pkl'
    csv_file = 'processed_data.csv'
    
    if os.path.exists(model_file) and os.path.exists(encoder_file) and os.path.exists(csv_file):
        try:
            with open(model_file, 'rb') as f:
                rf_model = pickle.load(f)
            with open(encoder_file, 'rb') as f:
                label_encoder = pickle.load(f)
            return rf_model, label_encoder, True  # Loaded from files
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            st.warning(f"Model load failed ({e}); training dynamically.")
    
    # Dynamic training if files missing/corrupted
    if not os.path.exists(csv_file):
        st.error("processed_data.csv missing! Add training data.")
        return None, None, False
    
    df = pd.read_csv(csv_file)
    if df.empty or 'quality_label' not in df.columns:
        st.error("CSV invalid: Needs 'quality_label' and features (word_count, flesch_reading_ease, etc.).")
        return None, None, False
    
    # Features (adjust columns to match your CSV)
    feature_cols = ['word_count', 'flesch_reading_ease', 'sentence_count', 'avg_sentence_length', 'thin_content']
    X = df[feature_cols].fillna(0)  # Handle NaNs
    y = df['quality_label']  # e.g., ['High', 'Medium', 'Low']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Balance if imbalanced (optional; remove if not needed)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y_encoded)
    
    # Train RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_balanced, y_balanced)
    
    # Save models (for next runs)
    with open(model_file, 'wb') as f:
        pickle.dump(rf_model, f)
    with open(encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    st.info("Models trained and saved dynamically.")
    return rf_model, label_encoder, False  # Trained dynamically

# Compute SEO features from text
def compute_features(text):
    # Download NLTK if needed
    download_nltk_data()
    
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)
    flesch_score = flesch_reading_ease(text)
    thin_content = word_count < 300  # Arbitrary threshold; adjust
    
    return {
        'word_count': word_count,
        'flesch_reading_ease': flesch_score,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'thin_content': thin_content
    }

# Fetch and clean content from URL
def fetch_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script/style
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text[:5000]  # Limit for processing
    except Exception as e:
        st.error(f"Failed to fetch {url}: {e}")
        return ""

# Compute SBERT similarity
@st.cache_resource
def load_sbert():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text, model):
    if not text.strip():
        return []
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    doc_embeddings = model.encode(sentences)
    ref_embeddings = model.encode(REFERENCE_SNIPPETS)
    similarities = cosine_similarity(doc_embeddings, ref_embeddings)
    top_indices = np.argsort(similarities.mean(axis=0))[::-1][:3]  # Top 3 refs
    similar_to = [REFERENCE_SNIPPETS[i] for i in top_indices]
    return similar_to

# Streamlit UI
def main():
    st.title("SEO Content Analyzer")
    st.write("Enter a URL to analyze content quality, readability, and SEO suggestions.")
    
    url = st.text_input("URL", value="https://example.com")
    
    if st.button("Analyze"):
        if not url.startswith(('http://', 'https://')):
            st.error("Please enter a valid URL starting with http:// or https://.")
            return
        
        with st.spinner("Fetching and analyzing content..."):
            text = fetch_content(url)
            if not text:
                return
            
            features = compute_features(text)
            sbert_model = load_sbert()
            similar_to = compute_similarity(text, sbert_model)
            
            # Prepare feature vector for prediction
            feature_vector = np.array([[features['word_count'], features['flesch_reading_ease'],
                                        features['sentence_count'], features['avg_sentence_length'],
                                        int(features['thin_content'])]])
            
            # Load/train models
            rf_model, label_encoder, from_file = load_or_train_models()
            if rf_model is None:
                st.error("Models failed to load/train. Check processed_data.csv.")
                return
            
            # Predict
            y_pred = rf_model.predict(feature_vector)[0]
            quality_label = label_encoder.inverse_transform([y_pred])[0]
            
            # Output JSON (as per your project)
            result = {
                "quality_label": quality_label,
                "word_count": features['word_count'],
                "flesch_reading_ease": round(features['flesch_reading_ease'], 1),
                "sentence_count": features['sentence_count'],
                "avg_sentence_length": round(features['avg_sentence_length'], 1),
                "thin_content": features['thin_content'],
                "similar_to": similar_to[:3]  # Limit to 3
            }
            
            st.json(result)
            st.success("Analysis complete!")

if __name__ == "__main__":
    main()
