import joblib
from utils.nlp_utils import clean_text  # Same as training script

def load_model_and_vectorizer():
    model = joblib.load("models/svm_model.pkl")  # Or use naive_bayes_model.pkl
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

def predict_category(resume_text):
    model, vectorizer = load_model_and_vectorizer()
    cleaned = clean_text(resume_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    sample_resume = """Experienced chartered accountant with 5 years of experience 
    in financial reporting, tax filing, and balance sheet auditing."""
    
    category = predict_category(sample_resume)
    print("Predicted Category:", category)
