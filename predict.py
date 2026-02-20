import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# User Input
email_text = input("Enter email message: ")

if not email_text.strip():
    print("Please enter a valid email message.")
    exit()

cleaned = preprocess(email_text)
vector_input = vectorizer.transform([cleaned]).toarray()

result = model.predict(vector_input)[0]
probability = model.predict_proba(vector_input)[0][1]

if result == 1:
    label = "Spam"
    risk = "HIGH RISK"
else:
    label = "Legitimate"
    risk = "LOW RISK"

print("\n===== Prediction Result =====")
print(f"Email Type       : {label}")
print(f"Fraud Probability: {round(probability*100,2)}%")
print(f"Risk Level       : {risk}")
print("=============================")
