import re
import spacy

nlp = spacy.load("en_core_web_sm")

# --- Basic Clean ---
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# --- Keyword Skill Extractor (custom) ---
def extract_skills(text):
    skills_keywords = [
        "accounting", "tax", "audit", "excel", "finance", "tally", "payroll", "budgeting",
        "reconciliation", "bookkeeping", "gst", "compliance"
    ]
    found_skills = [skill for skill in skills_keywords if skill in text.lower()]
    return list(set(found_skills))

# --- NER Extraction ---
def extract_entities(text):
    doc = nlp(text)
    orgs = list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    degrees = list(set(ent.text for ent in doc.ents if ent.label_ in ["EDUCATION", "DEGREE", "QUALIFICATION"]))
    return orgs, degrees

# --- NLP-based Smart Questions Generator ---
def generate_questions(skills, orgs=None, degrees=None):
    questions = []

    # Skill-based
    for skill in skills:
        questions.append(f"What is your experience with {skill}?")
        questions.append(f"Can you explain a project where you used {skill}?")
        questions.append(f"How do you stay updated on the latest trends in {skill}?")

    # Org-based
    if orgs:
        for org in orgs:
            questions.append(f"What was your role at {org}?")
            questions.append(f"What key contributions did you make at {org}?")

    # Degree-based
    if degrees:
        for degree in degrees:
            questions.append(f"Why did you choose to pursue {degree}?")
            questions.append(f"How has your education in {degree} helped you professionally?")

    return questions

