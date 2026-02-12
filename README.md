📧 Email Fraud Detection using NLP
1. Project Overview

This project builds a Machine Learning system to classify emails as Spam (Fraudulent) or Legitimate using Natural Language Processing techniques.

The model is trained on the SMS Spam Collection dataset and uses TF-IDF vectorization with a Multinomial Naive Bayes classifier.

2. Techniques Used

Text Preprocessing

Lowercasing

Punctuation removal

Stopword removal

Stemming

TF-IDF Feature Extraction

Multinomial Naive Bayes

Cross Validation

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

3. Model Performance

Accuracy: ~98%

Precision: 100%

Recall: ~85%

Cross Validation Accuracy: ~97–98%

The model achieves high precision, meaning legitimate emails are rarely misclassified as spam.

4. Project Structure
email-fraud-detection/
│
├── train.py
├── predict.py
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python train.py

3️⃣ Run prediction
python predict.py


Enter an email message when prompted.

5. Business Use Case

This system can be integrated into email services to automatically classify incoming messages, reducing spam and phishing risks.