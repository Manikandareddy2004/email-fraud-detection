import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.set_page_config(page_title="Email Fraud Detection")

st.title("📧 Email Fraud Detection System")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

email_text = st.text_area("Enter Email Message")

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = preprocess(email_text)
        vector_input = vectorizer.transform([cleaned]).toarray()
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0][1]

        if result == 1:
            st.error(f"⚠️ Spam (Fraud Probability: {round(probability*100,2)}%)")
        else:
            st.success(f"✅ Legitimate (Fraud Probability: {round(probability*100,2)}%)")
