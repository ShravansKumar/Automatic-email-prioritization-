
!pip install transformers
!pip install tensorflow
!pip install tensorflow-text

pip install --upgrade transformers
pip install --upgrade tensorflow

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# Load the dataset
df = pd.read_csv("pr.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Body"], df["priority"], test_size=0.2, random_state=42)

# Vectorize the email content using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classification model
classifier = LogisticRegression(multi_class="multinomial")
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
accuracy = classifier.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)

# Define the priority labels
priority_labels = ["low", "medium", "high"]

def predict_priority(emails):
    # Vectorize the emails using TF-IDF
    email_tfidf = vectorizer.transform(emails)

    # Predict the priority labels for the emails
    predictions = classifier.predict(email_tfidf)

    # Return the predicted labels as priorities
    return predictions

# Example usage
email_list = ["You have won a prize! Go to link to claim your $500 Amazon gift card","Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like spoilt child and he got caught up in that. Till 2! But we won't go there! Not doing too badly cheers. You? ","Your account is temporarily frozen. Please log in to to secure your account","Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's left over dinner ? Do you feel my Love ?"]
priorities = predict_priority(email_list)
print("Priorities:", priorities)