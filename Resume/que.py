import streamlit as st
import joblib
import io
from utils.pdf_utils import extract_text_from_pdf
from utils.nlp_utils import clean_text, extract_skills, extract_entities, generate_questions

# ---------------------
# Load model and vectorizer
# ---------------------
def load_model(model_name):
    model_path = f"models/{model_name}.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ---------------------
# Predict resume category
# ---------------------
def predict_category(text, model_name):
    model, vectorizer = load_model(model_name)
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    return prediction[0]

# ---------------------
# Streamlit App UI
# ---------------------
st.set_page_config(page_title="Resume Interview QG", layout="centered")
st.title("🤖 Resume-Based Question Generator")
st.markdown("Upload a resume PDF, choose a model, and get smart interview questions based on its content.")

# Model dropdown: SVM or Naive Bayes
model_option = st.selectbox(
    "Choose a classification model:",
    options=["svm_model", "naive_bayes_model"],
    format_func=lambda x: "SVM" if x == "svm_model" else "Naive Bayes"
)

# Upload resume PDF
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

# ---------------------
# Process resume
# ---------------------
if uploaded_file:
    with st.spinner("🔍 Processing resume..."):
        # Extract text from uploaded PDF
        text = extract_text_from_pdf(uploaded_file)

        # Predict category
        category = predict_category(text, model_option)

        # Clean and extract additional info
        cleaned = clean_text(text)
        skills = extract_skills(cleaned)
        orgs, degrees = extract_entities(text)
        questions = generate_questions(skills, orgs, degrees)

        # Keep only top 15 questions
        questions = questions[:15]

        # Prepare questions as text for download
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    # ---------------------
    # Display results
    # ---------------------
    st.success("✅ Resume processed successfully!")

    st.subheader("📌 Predicted Category:")
    st.markdown(f"**{category}**")

    st.subheader("🧠 Extracted Skills:")
    st.markdown(", ".join(skills) if skills else "_None found_")

    st.subheader("🏢 Organizations Detected:")
    st.markdown(", ".join(orgs) if orgs else "_None found_")

    st.subheader("🎓 Degrees Detected:")
    st.markdown(", ".join(degrees) if degrees else "_None found_")

    st.subheader("💬 Top 15 Interview Questions:")
    for i, q in enumerate(questions, 1):
        st.markdown(f"{i}. {q}")

    # ---------------------
    # Download Button
    # ---------------------
    st.subheader("📥 Download Questions:")
    st.download_button(
        label="Download Interview Questions (.txt)",
        data=questions_text,
        file_name="interview_questions.txt",
        mime="text/plain"
    )
