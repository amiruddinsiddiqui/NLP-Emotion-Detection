# 🧠 Emotion Classifier

This project predicts human emotions from text using **NLP (Natural Language Processing)**.  
It classifies text into six emotions — *sadness, anger, love, surprise, fear,* and *joy* — using a **Logistic Regression model with Bag of Words (BoW) vectorization**.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://nlp-emotion-detection-a.streamlit.app/)

---

## ⚙️ Tech Stack
- Python 3.13.5  
- scikit-learn 1.6.1  
- numpy 2.1.3  
- pandas 2.2.3  
- joblib 1.4.2  
- Streamlit

---

## 🚀 Run Locally
```bash
python -m venv emotion_env
emotion_env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Performance
| Model | Accuracy |
|--------|-----------|
| BoW + Logistic Regression | **0.8887** |
| TF-IDF + Logistic Regression | 0.8615 |
| BoW + Naive Bayes | 0.7678 |
| TF-IDF + Naive Bayes | 0.6609 |

✅ **Best Model:** Logistic Regression (BoW)

---

## 📁 Files
- `app.py` → Streamlit web app  
- `LogRegEmo.pkl` → Trained model  
- `bow_vectorizer.pkl` → BoW vectorizer  
- `Untitled.ipynb` → Training notebook  

---

## 🧩 Conclusion
Among all models tested, **BoW + Logistic Regression** gave the highest accuracy of **0.8887**, showing that simple NLP techniques can effectively classify emotions from text.
