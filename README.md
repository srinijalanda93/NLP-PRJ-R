# 🤖 Resume-Based Question Generator

A smart NLP-powered system that reads a candidate's resume (in PDF), predicts their job category using Machine Learning models, extracts key details, and generates personalized interview questions — all in one place using an interactive Streamlit web app.

---

## 🔍 Theme / Domain
**Machine Learning Models + Natural Language Processing**

The project combines classical machine learning, text mining, and resume parsing to build an intelligent interview assistant.

---

## 🎯 Aim
To develop an automated system that:
- Predicts the job category of a candidate based on resume content
- Extracts structured details like skills, education, and experience
- Generates intelligent, personalized interview questions tailored to the candidate's background

---
##Dataset 
Used kaggle dataset  <a link="https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset">Kaggle</a>
and data preprocessing is in

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
NLP-PRJ-R/
│ ├── Resume/ 
  │ ├── models/ # Trained ML models + vectorizer 
    │ │ ├── naive_bayes_model.pkl
    │ │ ├── svm_model.pkl
    │ │ └── tfidf_vectorizer.pkl 
│ ├── sample_resumes/ 
    │ │ └── ACCOUNTANT/ # Sample PDF resumes for testing
│ ├── utils/ # Utility scripts 
      │ │ ├── nlp_utils.py # Text cleaning, entity extraction, question generation 
      │ │ └── pdf_utils.py # PDF text extraction using PyMuPDF 
 ├── app.py # Main Streamlit app for UI
 ├── predict_resume_category.py # Script for backend prediction logic
 ├── que.py # Standalone script for question generation 
 ├── train_accountant_model.py # Script to train SVM / Naive Bayes on labeled resumes
 ├── finance_accountant_resumes.csv # Resume dataset used for training
 ├── accuracy values.txt # Evaluation results of models 
 ├── requirements.txt # Python dependencies 
 ├── README.md 
<img width="308" alt="Screenshot 2025-04-08 at 12 18 48 PM" src="https://github.com/user-attachments/assets/2287567c-623b-4cc8-812d-81721c5f0220" />


 <h1>Frontent part using streamlit</h1>
 <h2>USING SVM </h2>
 <img width="1291" alt="Screensh<img width="472" alt="Screenshot 2025-04-08 at 12 51 59 PM" src="https://github.com/user-attachments/assets/eb53aa1b-58a1-4b66-99a7-889906ea7310" />

 
<img width="1291" alt="Screenshot 2025-04-08 at 12 50 04 PM" src="https://github.com/user-attachments/assets/a417f20a-1452-4e3b-a6c7-2e8f28502cf3" />
<h2>Using navie base</h2>
<img width="472" alt="Screenshot 2025-04-08 at 12 51 59 PM" src="https://github.com/user-attachments/assets/33fc1320-8263-4a13-b893-771ffda27e25" />



<img width="472" alt="Screens<img width="718" alt="Screenshot 2025-04-08 at 1 09 40 PM" src="https://github.com/user-attachments/assets/ab60b5a8-eb33-445f-b256-26da6c5afe0a" />
<img width="718" alt="Screenshot 2025-04-08 at 1 10 23 PM" src="https://github.com/user-attachments/assets/4847bc4e-e768-4f8b-b776-0adb92a16301" />



