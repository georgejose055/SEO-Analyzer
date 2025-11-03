import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import textstat
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import pickle
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# Disable parallelism for Streamlit Cloud
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="SEO Analyzer", page_icon="🔍", layout="wide")

@st.cache_resource
def load_or_train_rf():
    """
    Load RF model or fallback to training from processed_data.csv.
    Features: word_count, flesch_reading_ease, sentence_count, avg_sentence_length.
    Labels: Low, Medium, High (from dataset).
    """
    try:
        # If .pkl exists (optional with LFS)
        rf_model = joblib.load('rf_model.pkl')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return rf_model, label_encoder
    except FileNotFoundError:
        # Fallback: Train small RF (fast on 69 rows)
        df = pd.read_csv('processed_data.csv')  # Assumes columns: word_count, flesch_reading_ease, sentence_count, avg_sentence_length, quality_label
        features = ['word_count', 'flesch_reading_ease', 'sentence_count', 'avg_sentence_length']
        X = df[features].fillna(0).values
        le = LabelEncoder()
        y = le.fit_transform(df['quality_label'])  # Fit on Low, Medium, High
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        return rf_model, le

@st.cache_resource
def load_sbert():
    """Load SBERT dynamically (downloads/caches ~80MB on first run)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load models
rf_model, label_encoder = load_or_train_rf()
sbert_model = load_sbert()

# Load dataset for similarity (preprocess sentences)
@st.cache_data
def load_dataset_sentences():
    """Load processed_data.csv sentences for similarity comparison."""
    df = pd.read_csv('processed_data.csv')
    # Assume 'sentences' column or derive; placeholder: split text if not
    all_sentences = []
    for idx, row in df.iterrows():
        # If 'text' column exists, split; else use placeholder
        text = row.get('text', 'Sample text for similarity.')  # Adjust column name
        sentences = nltk.sent_tokenize(text)
        all_sentences.extend(sentences)
    return all_sentences

dataset_sentences = load_dataset_sentences()

def fetch_and_extract_text(url):
    """Fetch URL content with User-Agent, extract clean text."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script/style
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        st.error(f"Fetch error: {e}")
        return ""

def compute_features(text):
    """Compute SEO features."""
    if not text:
        return {}
    word_count = len(text.split())
    sentences = nltk.sent_tokenize(text)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    return {
        "word_count": word_count,
        "flesch_reading_ease": flesch_reading_ease,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length
    }

def predict_quality(features):
    """Predict quality label with RF."""
    feature_cols = ['word_count', 'flesch_reading_ease', 'sentence_count', 'avg_sentence_length']
    X = np.array([[features.get(col, 0) for col in feature_cols]]).reshape(1, -1)
    label_idx = rf_model.predict(X)[0]
    quality_label = label_encoder.inverse_transform([label_idx])[0]
    return quality_label

def compute_similarity(text):
    """Compute similar sentences from dataset using SBERT (cosine >0.85)."""
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []
    query_embs = sbert_model.encode(sentences)
    dataset_embs = sbert_model.encode(dataset_sentences)
    similarities = cosine_similarity(query_embs, dataset_embs)
    similar_indices = np.where(similarities > 0.85)[1]
    similar_urls = [dataset_sentences[i][:50] + '...' for i in similar_indices]  # Placeholder; map to URLs if in df
    return list(set(similar_urls))  # Unique

# UI
st.title("🔍 SEO Analyzer")
st.markdown("Enter a URL to analyze SEO quality, readability, thin content, and content similarity.")

url = st.text_input("URL:", placeholder="https://en.wikipedia.org/wiki/Search_engine_optimization")

if st.button("Analyze SEO", type="primary"):
    if url:
        with st.spinner("Fetching and analyzing... (may take 10-30s for embeddings)"):
            text = fetch_and_extract_text(url)
            if text:
                features = compute_features(text)
                quality_label = predict_quality(features)
                thin_content = features["word_count"] < 300
                similar_to = compute_similarity(text)
                
                result = {
                    "quality_label": quality_label,
                    "word_count": features["word_count"],
                    "flesch_reading_ease": round(features["flesch_reading_ease"], 2),
                    "sentence_count": features["sentence_count"],
                    "avg_sentence_length": round(features["avg_sentence_length"], 2),
                    "thin_content": thin_content,
                    "similar_to": similar_to[:5]  # Top 5
                }
                
                st.success("Analysis complete!")
                st.json(result)
                
                # Metrics display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Quality", quality_label)
                with col2:
                    st.metric("Word Count", features["word_count"])
                with col3:
                    st.metric("Readability Score", round(features["flesch_reading_ease"], 2))
                
                if thin_content:
                    st.warning("Thin content detected (under 300 words) – improve depth!")
                if similar_to:
                    st.info(f"Similar content found: {len(similar_to)} matches.")
            else:
                st.error("No text extracted. Check URL or site blocks scraping.")
    else:
        st.warning("Please enter a valid URL.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, scikit-learn, and Sentence Transformers. Data from Kaggle SEO dataset.")
