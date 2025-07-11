import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# ✅ Load directly from files in root folder
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 1  # Fake
true['label'] = 0  # Real

# Combine and shuffle
df = pd.concat([fake, true], ignore_index=True)
df = df[['text', 'label']].sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("backend/model", exist_ok=True)
dump(model, "backend/model/fake_news_model.pkl")
dump(vectorizer, "backend/model/vectorizer.pkl")

print("✅ Model and vectorizer saved to backend/model/")
