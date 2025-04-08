import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils.nlp_utils import clean_text

# 1. Load and clean data
df = pd.read_csv("/Users/srinija/Desktop/Resume/finance_accountant_resumes.csv")
df["cleaned"] = df["Resume_str"].apply(clean_text)

X = df["cleaned"]
y = df["Category"]

# 2. Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train models
nb_model = MultinomialNB()
svm_model = LinearSVC()

nb_model.fit(X_train_vec, y_train)
svm_model.fit(X_train_vec, y_train)

# 5. Evaluate models
print("\n--- Naive Bayes Evaluation ---")
nb_preds = nb_model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))

print("\n--- SVM Evaluation ---")
svm_preds = svm_model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

# 6. Save models
joblib.dump(nb_model, "models/naive_bayes_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print(" Models trained, evaluated, and saved.")
