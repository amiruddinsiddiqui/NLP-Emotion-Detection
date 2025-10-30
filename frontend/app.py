import streamlit as st
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "LogRegEmo.pkl")
vectorizer_path = os.path.join(BASE_DIR, "bow_vectorizer.pkl")

# Load the model and BoW vectorizer
try:
    model = joblib.load(model_path)
    bow_vec = joblib.load(vectorizer_path)
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Emotion mapping
emotion_labels = {
    0: "Sadness 😢",
    1: "Anger 😡",
    2: "Love ❤️",
    3: "Surprise 😲",
    4: "Fear 😨",
    5: "Joy 😊"
}

# Streamlit ui
st.set_page_config(page_title="Emotion Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 Emotion Detection App")
st.write("Type your text below to predict the emotion:")

# Text input
user_input = st.text_area("Enter a sentence", placeholder="e.g. I'm feeling amazing today!")

if st.button("Analyze Emotion"):
    if user_input.strip():
        # Transform and predict
        text_vector = bow_vec.transform([user_input])
        prediction = model.predict(text_vector)[0]

        # Show result
        st.success(f"**Predicted Emotion:** {emotion_labels[prediction]}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("Model: Logistic Regression | Vectorizer: Bag of Words (BoW)")
