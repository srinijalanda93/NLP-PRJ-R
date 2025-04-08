# 🤖 Resume-Based Question Generator

A smart NLP-powered system that reads a candidate's resume (in PDF), predicts their job category using Machine Learning models, extracts key details, and generates personalized interview questions — all in one place using an interactive Streamlit web app.

---

## 🔍 Theme / Domain
**Artificial Intelligence + Natural Language Processing**

The project combines classical machine learning, text mining, and resume parsing to build an intelligent interview assistant.

---

## 🎯 Aim
To develop an automated system that:
- Predicts the job category of a candidate based on resume content
- Extracts structured details like skills, education, and experience
- Generates intelligent, personalized interview questions tailored to the candidate's background

---

## 💡 Problem Definition

Manual resume screening and question preparation are time-consuming and inconsistent. Recruiters often need to:
- Skim through large volumes of resumes
- Identify relevant skills and qualifications
- Prepare contextually appropriate interview questions

This project solves that by automating:
- Resume parsing from PDFs
- Classification into job categories
- NLP-based skill extraction
- Smart question generation for interviews

---

## 🚀 Outcome / Deliverables

- 🧠 Trained SVM & Naive Bayes models for job category prediction using TF-IDF
- 📄 PDF resume parser using PyMuPDF
- 🛠️ NLP utilities to clean, extract, and transform text
- 💬 Question generator based on detected skills, degrees, and organizations
- 🌐 Streamlit web app for uploading, predicting, displaying, and downloading questions

---

## 🧰 Tech Stack

| Area                | Tools / Libraries Used            |
|---------------------|----------------------------------|
| Programming Language| Python 3.x                        |
| NLP / ML            | Scikit-learn, NLTK, Regex         |
| Resume Parsing      | PyMuPDF (`fitz`)                  |
| Vectorization       | TF-IDF (from `sklearn`)           |
| Web UI              | Streamlit                         |
| Model Storage       | joblib                            |

---

## 🧱 Project Structure
NLP-PRJ-R/ │ ├── Resume/ │ ├── models/ # Trained ML models + vectorizer │ │ ├── naive_bayes_model.pkl │ │ ├── svm_model.pkl │ │ └── tfidf_vectorizer.pkl │ │ │ ├── sample_resumes/ │ │ └── ACCOUNTANT/ # Sample PDF resumes for testing │ │ │ ├── utils/ # Utility scripts │ │ ├── nlp_utils.py # Text cleaning, entity extraction, question generation │ │ └── pdf_utils.py # PDF text extraction using PyMuPDF │ ├── app.py # Main Streamlit app for UI ├── predict_resume_category.py # Script for backend prediction logic ├── que.py # Standalone script for question generation ├── train_accountant_model.py # Script to train SVM / Naive Bayes on labeled resumes ├── finance_accountant_resumes.csv # Resume dataset used for training ├── accuracy values.txt # Evaluation results of models ├── requirements.txt # Python dependencies ├── README.md 
