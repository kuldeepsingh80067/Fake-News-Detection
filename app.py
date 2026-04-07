# ==========================================
# 📰 Fake News Detection PRO (Enhanced UI)
# Author: Kuldeep Singh
# ==========================================

import streamlit as st
import numpy as np
import re
from PIL import Image
import requests
from bs4 import BeautifulSoup

# OPTIONAL OCR
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
except:
    reader = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# ⚙️ PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Fake News Detector PRO", page_icon="📰")

# ==========================================
# 🧹 TEXT CLEANING
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ==========================================
# 🤖 MODEL (UNCHANGED)
# ==========================================
@st.cache_resource
def train_model():
    texts = [
        "government releases official report on economy growth",
        "scientists confirm new discovery in space research",
        "india wins cricket match against australia in final",
        "new education policy announced by government",
        "nasa launches new satellite successfully",

        "you wont believe what happened next shocking truth",
        "click here to earn money instantly from home",
        "miracle cure doctors dont want you to know",
        "secret trick to become rich overnight revealed",
        "breaking shocking news celebrity scandal leaked"
    ]

    labels = [1,1,1,1,1, 0,0,0,0,0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer

model, vectorizer = train_model()

# ==========================================
# 🧠 RULE CHECK (UNCHANGED)
# ==========================================
def rule_based_check(text):
    fake_keywords = [
        "shocking", "click", "earn money", "miracle",
        "secret", "doctors hate", "100% guarantee",
        "instant", "overnight"
    ]
    score = sum(word in text for word in fake_keywords)
    return score

# ==========================================
# 🌐 URL EXTRACT
# ==========================================
def get_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except:
        return ""

# ==========================================
# 🖼️ OCR
# ==========================================
def extract_text_from_image(image):
    if reader is None:
        return ""
    result = reader.readtext(np.array(image))
    return " ".join([r[1] for r in result])

# ==========================================
# 🎨 UI
# ==========================================
st.sidebar.title("👨‍💻 About")
st.sidebar.markdown("**Developer:** Kuldeep Singh 🚀")

st.title("📰 Fake News Detection PRO")
st.markdown("Now supports **Text, URL, Image & Camera** 🔥")

option = st.radio("Choose Input Type:", ["Text", "URL", "Image Upload", "Camera"])

news_text = ""

# TEXT
if option == "Text":
    news_text = st.text_area("✍️ Enter News Text")

# URL
elif option == "URL":
    url = st.text_input("🔗 Enter News URL")
    if url:
        with st.spinner("Extracting..."):
            news_text = get_text_from_url(url)
        if news_text:
            st.success("✅ Text extracted")

# IMAGE UPLOAD
elif option == "Image Upload":
    file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        with st.spinner("Reading text..."):
            news_text = extract_text_from_image(image)
        if news_text:
            st.success("✅ Text extracted")

# CAMERA
elif option == "Camera":
    camera = st.camera_input("📸 Click Photo")
    if camera:
        image = Image.open(camera)
        st.image(image, use_column_width=True)
        with st.spinner("Reading text..."):
            news_text = extract_text_from_image(image)
        if news_text:
            st.success("✅ Text extracted")

# ==========================================
# 🔍 PREDICTION (UNCHANGED)
# ==========================================
if st.button("🔍 Analyze News"):
    if not news_text.strip():
        st.warning("⚠️ Provide input")
    else:
        cleaned = clean_text(news_text)

        vectorized = vectorizer.transform([cleaned])
        ml_pred = model.predict(vectorized)[0]
        confidence = np.max(model.predict_proba(vectorized))

        rule_score = rule_based_check(cleaned)

        if rule_score >= 2:
            final_pred = 0
            confidence = max(confidence, 0.75)
        else:
            final_pred = ml_pred

        st.subheader("📊 Result")

        if final_pred == 1:
            st.success(f"✅ Real News ({confidence*100:.2f}%)")
        else:
            st.error(f"❌ Fake News ({confidence*100:.2f}%)")

        if rule_score > 0:
            st.info("⚠️ Contains suspicious words")

# ==========================================
# 📌 FOOTER
# ==========================================
st.markdown("---")
st.markdown("### 👨‍💻 Developed by Kuldeep Singh")
