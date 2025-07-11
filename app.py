import streamlit as st
from detector import detect_fake_news
from generator import generate_fake_news

st.set_page_config(page_title="📰 Fake News Generator & Detector", layout="centered")

st.title("📰 Fake News Generator & Detector")
st.markdown("Detect or Generate news using AI and NLP.")

tab1, tab2 = st.tabs(["🔍 Detector", "🎭 Generator"])

with tab1:
    st.subheader("Detect Fake News")
    news_input = st.text_area("Enter news content:", height=200)
    if st.button("Detect"):
        if news_input.strip():
            result = detect_fake_news(news_input)
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter some text.")

with tab2:
    st.subheader("Generate Fake News")
    prompt = st.text_input("Enter a prompt (e.g., 'Breaking News:')", "Breaking News:")
    if st.button("Generate"):
        fake_news = generate_fake_news(prompt)
        st.text_area("Generated Fake News", value=fake_news, height=200)
