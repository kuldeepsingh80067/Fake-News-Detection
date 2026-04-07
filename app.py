# ==========================================
# 📰 Fake News Detection PRO (No Setup Version)
# Author: Kuldeep Singh
# ==========================================

import streamlit as st
import numpy as np
import re
import string

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
# 🤖 QUICK TRAIN MODEL (INSTANT)
# ==========================================
@st.cache_resource
def train_model():
    texts = [
        # REAL NEWS
        "government releases official report on economy growth",
        "scientists confirm new discovery in space research",
        "india wins cricket match against australia in final",
        "new education policy announced by government",
        "nasa launches new satellite successfully",

        # FAKE NEWS
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
# 🧠 SMART RULE CHECK (BOOST ACCURACY)
# ==========================================
def rule_based_check(text):
    fake_keywords = [
        "shocking", "click", "earn money", "miracle",
        "secret", "doctors hate", "100% guarantee",
        "instant", "overnight"
    ]

    score = 0
    for word in fake_keywords:
        if word in text:
            score += 1

    return score

# ==========================================
# 🎨 UI
# ==========================================
st.sidebar.title("👨‍💻 About")
st.sidebar.markdown("""
**Developer:** Kuldeep Singh  
AI/ML Enthusiast 🚀  
""")

st.title("📰 Fake News Detection PRO")
st.markdown("Detect Fake vs Real news using AI 🤖")

news_text = st.text_area("✍️ Enter News Text")

# ==========================================
# 🔍 PREDICTION
# ==========================================
if st.button("🔍 Analyze News"):

    if not news_text.strip():
        st.warning("⚠️ Please enter news text")
    else:
        cleaned = clean_text(news_text)

        # ML prediction
        vectorized = vectorizer.transform([cleaned])
        ml_pred = model.predict(vectorized)[0]
        confidence = np.max(model.predict_proba(vectorized))

        # Rule-based boost
        rule_score = rule_based_check(cleaned)

        # FINAL DECISION (HYBRID 🔥)
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

        # Explanation
        if rule_score > 0:
            st.info("⚠️ Detected sensational or spam-like words")

# ==========================================
# 📌 FOOTER
# ==========================================
st.markdown("---")
st.markdown("### 👨‍💻 Developed by Kuldeep Singh")
st.caption("Built with ❤️ using AI, ML & Streamlit")
