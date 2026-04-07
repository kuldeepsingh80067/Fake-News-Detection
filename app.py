# ==========================================
# 📰 Fake News Detection App (Streamlit)
# Author: Kuldeep Singh
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 🧹 TEXT CLEANING FUNCTION
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# ==========================================
# 📂 LOAD OR TRAIN MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    except:
        # If no saved model, train a simple one
        data = {
            "text": [
                "Breaking: Government announces new scheme",
                "You won't believe this shocking news!!!",
                "Official report confirms economic growth",
                "Fake miracle cure discovered!!!",
                "Scientists release verified climate report",
                "Click here to earn money fast!!!"
            ],
            "label": [1, 0, 1, 0, 1, 0]
        }

        df = pd.DataFrame(data)
        df["text"] = df["text"].apply(clean_text)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]

        model = LogisticRegression()
        model.fit(X, y)

        pickle.dump(model, open("model.pkl", "wb"))
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================
# 🎨 STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.markdown("Detect whether a news article is **Real or Fake** using AI 🤖")

# Input
user_input = st.text_area("✍️ Enter News Text Here:")

# Button
if st.button("🔍 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()

        st.subheader("📊 Result:")

        if prediction == 1:
            st.success(f"✅ This looks like REAL news")
        else:
            st.error(f"❌ This looks like FAKE news")

        st.write(f"Confidence: {confidence*100:.2f}%")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")