import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize
import textstat
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import time
nltk.download('punkt', quiet=True)

# Selenium (optional; local only)
USE_SELENIUM = st.sidebar.checkbox("Use Selenium (Local Anti-Block)", value=False)  # Disable for deploy
if USE_SELENIUM:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        st.success("Selenium ready (slow but bypasses blocks).")
    except ImportError:
        st.error("Install: pip install selenium webdriver-manager")
        USE_SELENIUM = False

@st.cache_resource
def load_models():
    try:
        rf = joblib.load('rf_model.pkl')
        le = joblib.load('label_encoder.pkl')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return rf, le, model, le.classes_
    except FileNotFoundError:
        st.error("Models missing. Run notebook Cell 8.")
        return None, None, None, None

global_rf, global_le, global_model, global_label_names = load_models()

@st.cache_data
def load_embeddings():
    try:
        df = pd.read_csv('processed_data.csv')
        texts = df['text'].tolist()
        urls = df['url'].tolist()
        embeddings = global_model.encode(texts) if global_model else None
        return embeddings, urls, texts
    except FileNotFoundError:
        st.warning("No processed_data.csv; dups disabled.")
        return None, None, None

global_embeddings, global_urls, global_texts = load_embeddings()

def find_similar(text, threshold=0.85):
    if global_embeddings is None or global_model is None:
        return []
    emb = global_model.encode([text])
    sims = cosine_similarity(emb, global_embeddings)[0]
    return [global_urls[i] for i, sim in enumerate(sims) if sim > threshold and sim < 1.0][:3]

def selenium_fetch(url, timeout=60):
    if not USE_SELENIUM:
        return None
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(3)  # Wait for JS
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        st.error(f"Selenium error: {str(e)} (Try manual).")
        return None

def fetch_html(url, timeout=30, proxy=None):
    if USE_SELENIUM:
        return selenium_fetch(url, timeout)
    try:
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        session = requests.Session()  # Cookies
        headers = {
            'User-Agent': np.random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }
        session.headers.update(headers)
        time.sleep(2)  # Anti-rate
        resp = session.get(url, timeout=timeout, proxies=proxy)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            st.error(f"403 Block: {url}. Enable Selenium or use Manual (browser source). Proxies help.")
        elif e.response.status_code == 429:
            st.error("Rate-limited. Wait/retry.")
        else:
            st.error(f"HTTP: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Fetch: {str(e)}")
        return None

def extract_title_body(html):
    if not html:
        return "No title", ""
    soup = BeautifulSoup(html, 'html.parser')
    title = (soup.title.string.strip() if soup.title and soup.title.string else "No title")
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'sidebar']):
        tag.decompose()
    body = soup.get_text(separator=' ', strip=True)
    return title, body

def compute_features(text):
    if not text or len(text.strip()) < 20:
        return 0, 0, 0, 0
    text = re.sub(r'\s+', ' ', text.strip().lower())
    words = re.findall(r'\b\w+\b', text)
    sentences = [s for s in sent_tokenize(text) if len(re.findall(r'\b\w+\b', s)) > 1]
    word_count = len(words)
    sentence_count = len(sentences)
    try:
        raw_flesch = textstat.flesch_reading_ease(text[:4000])
    except:
        raw_flesch = np.nan
    avg_sent_len = word_count / max(sentence_count, 1)
    
    def proxy_flesch(t):
        w = re.findall(r'\b\w+\b', t.lower())
        if len(w) == 0:
            return 40.0
        sents = sent_tokenize(t)
        long_s = [s for s in sents if len(re.findall(r'\b\w+\b', s.lower())) > 3]
        adj_sent = len(long_s) or 1
        avg_s = len(w) / adj_sent
        syl = sum(len(re.findall(r'[aeiouy]', ww.lower())) for ww in w) / len(w)
        score = 206.835 - 1.015 * avg_s - 84.6 * syl
        return max(30, min(120, score))
    
    flesch = proxy_flesch(text) if np.isnan(raw_flesch) or raw_flesch < 0 else raw_flesch
    return word_count, sentence_count, flesch, avg_sent_len

def get_quality_label_and_conf(pred, confidence, word_count, flesch):
    pred = int(pred[0]) if isinstance(pred, np.ndarray) else int(pred)
    label = global_label_names[pred] if global_label_names is not None else 'Low'
    if confidence < 0.65 and global_label_names is not None:
        if word_count > 1800 and flesch > 45:
            label = 'Medium'
        elif word_count > 350 and flesch > 40:
            label = 'Medium'
        else:
            label = 'Low'
    return label, confidence

def analyze_url(url, manual_html=None, proxy=None):
    if manual_html:
        html = manual_html
        st.info(f"Manual HTML ({len(html)} chars)")
    else:
        html = fetch_html(url, proxy=proxy)
        if not html:
            return {"url": url, "error": "Fetch failed"}
    
    title, body = extract_title_body(html)
    word_count, sentence_count, flesch, avg_sent_len = compute_features(body)
    if global_rf is None:
        return {"url": url, "error": "Model not loaded"}
    features = np.array([[word_count, sentence_count, flesch, avg_sent_len]])
    pred = global_rf.predict(features)[0]
    probs = global_rf.predict_proba(features)[0]
    confidence = np.max(probs)
    
    quality_label, adj_conf = get_quality_label_and_conf(pred, confidence, word_count, flesch)
    is_thin = word_count < 350 or sentence_count < 8
    similar_to = find_similar(body)
    
    result = {
        "url": url,
        "title": title[:80] + "..." if len(title) > 80 else title,
        "word_count": int(word_count),
        "readability": round(flesch, 1),
        "quality_label": quality_label,
        "confidence": round(adj_conf, 2),
        "is_thin": is_thin,
        "similar_to": similar_to
    }
    return result

@st.cache_data
def _display_results(result):
    st.success("Complete!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", result["word_count"])
    with col2:
        st.metric("Readability", f"{result['readability']:.1f}")
    with col3:
        st.metric("Confidence", f"{result['confidence']:.2f}")
    st.json(result)
    st.write("Thin" if result["is_thin"] else "Not Thin")
    if result["similar_to"]:
        st.subheader("Dups:")
        for u in result["similar_to"]:
            st.write(u)
    else:
        st.info("No dups.")

# UI (after all defs)
st.title("🕷️ SEO Content Quality & Duplicate Detector")
st.markdown("Analyzer with anti-block fetch. Repo: [GitHub](https://github.com/yourusername/seo-content-quality-detector)")

# Proxy Sidebar (run early for config)
with st.sidebar:
    proxy_url = st.text_input("Proxy (http://user:pass@ip:port):", placeholder="Optional for blocks")
    proxy = {'http': proxy_url, 'https': proxy_url} if proxy_url else None
    st.info("Proxy helps bypass IP blocks (e.g., free from proxyscrape.com).")
    st.header("Model")
    st.markdown("- RF (88% acc)\n- SBERT dups")
    tests = {"Example (Low)": "https://example.com", "Wiki (Med)": "https://en.wikipedia.org/wiki/Search_engine_optimization", "ResearchGate Test": "https://www.researchgate.net/publication/337039884_An_Overview_of_Dyslexia_Definition_Characteristics_Assessment_Identification_and_Intervention", "Quotes (Med)": "http://quotes.toscrape.com/"}
    selected = st.selectbox("Test:", list(tests.keys()))
    if st.button("Test"):
        test_url = tests[selected]
        result = analyze_url(test_url, proxy=proxy)
        if "error" not in result:
            _display_results(result)

input_type = st.radio("Input:", ["URL (Auto-Fetch)", "Manual HTML"])

if input_type == "URL (Auto-Fetch)":
    url = st.text_input("URL:", placeholder="https://example.com")
    manual_html = None
    if st.button("Analyze", type="primary"):
        if not url.startswith(('http://', 'https://')):
            st.warning("Valid URL (http(s)://).")
        else:
            with st.spinner("Analyzing..."):
                result = analyze_url(url, proxy=proxy)
                if "error" in result:
                    st.error(result["error"])
                else:
                    _display_results(result)
else:
    manual_html = st.text_area("HTML:", placeholder="<!DOCTYPE html>...", height=300)
    url = "manual"
    if st.button("Analyze", type="primary"):
        if not manual_html.strip():
            st.warning("Paste HTML.")
        else:
            with st.spinner("Analyzing..."):
                result = analyze_url(url, manual_html)
                if "error" in result:
                    st.error(result["error"])
                else:
                    _display_results(result)

st.markdown("---")
st.markdown("Streamlit | Kaggle | scikit-learn. For blocks: Manual/Selenium/Proxies/APIs.")
