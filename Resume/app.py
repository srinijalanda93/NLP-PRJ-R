import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from utils.nlp_utils import clean_text, extract_skills, extract_entities, generate_questions
import joblib
import os
import io
#Resume Text → [Clean] → [TF-IDF Vectorizer] → [ML Model (SVM or Naive Bayes)] → Predicted Category
# Function to load selected model and vectorizer
def load_model(model_name):
    model_path = f"models/{model_name}.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Predict category using chosen model
def predict_category(text, model_name):
    model, vectorizer = load_model(model_name)
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    return prediction[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Interview QG", layout="centered")
st.title("🤖 Resume-Based Question Generator")
st.markdown("Upload a resume PDF, choose a model, and get smart interview questions based on its content.")

# Model selection dropdown
model_option = st.selectbox(
    "Choose a classification model:",
    options=["svm_model", "naive_bayes_model"],
    format_func=lambda x: "SVM" if x == "svm_model" else "Naive Bayes"
)

# Resume upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Processing resume..."):
        text = extract_text_from_pdf(uploaded_file)
        category = predict_category(text, model_option)
        cleaned = clean_text(text)
        skills = extract_skills(cleaned)
        orgs, degrees = extract_entities(text)
        questions = generate_questions(skills, orgs, degrees)

    st.success("✅ Resume processed successfully!")

    # Display results
    st.subheader("📌 Predicted Category:")
    st.markdown(f"**{category}**")

    st.subheader("🧠 Extracted Skills:")
    st.markdown(", ".join(skills) if skills else "_None found_")

    st.subheader("🏢 Organizations Detected:")
    st.markdown(", ".join(orgs) if orgs else "_None found_")

    st.subheader("🎓 Degrees Detected:")
    st.markdown(", ".join(degrees) if degrees else "_None found_")

    st.subheader("💬 Interview Questions:")
    for i, q in enumerate(questions, 1):
        st.markdown(f"{i}. {q}")
