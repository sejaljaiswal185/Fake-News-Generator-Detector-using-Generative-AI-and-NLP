from joblib import load

model = load("backend/model/fake_news_model.pkl")
vectorizer = load("backend/model/vectorizer.pkl")

def detect_fake_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Fake News 🟥" if prediction == 1 else "Real News 🟩"
