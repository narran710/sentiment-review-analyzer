import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("🎭 Sentiment Review Analyzer")
st.write("Analyze text sentiment using a fine-tuned BERT model.")

user_input = st.text_area("Enter your review text:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        payload = {"text": user_input}
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            sentiment = result["sentiment"]

            if sentiment == "positive":
                st.success(f"💚 Sentiment: **{sentiment.upper()}**")
            else:
                st.error(f"💔 Sentiment: **{sentiment.upper()}**")
        else:
            st.error("API Error: Could not get response.")
