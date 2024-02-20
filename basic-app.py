from fastapi import FastAPI
import uvicorn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# Creating FastAPI instance
app = FastAPI()

origins = [
    "http://localhost:5175",  # Replace with your Vite app's origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST1"],
    allow_headers=["*"],
)
# Creating class to define the request body
# and the type hints of each attribute


class request_body(BaseModel):
    email_text: str


# Loading Iris Dataset
df = pd.read_csv("pr.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["Body"], df["priority"], test_size=0.2, random_state=42)

# Vectorize the email content using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classification model
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
accuracy = classifier.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy6)

# Define the priority labels
priority_labels = ["low", "medium", "high"]

# Creating an Endpoint to receive the data
# to make prediction on.


@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    test_data = [
        data.email_text,
    ]
    # Vectorize the emails using TF-IDF
    email_tfidf = vectorizer.transform(test_data)

    # Predict the priority labels for the emails
    predictions = classifier.predict(email_tfidf)

    # Return the Result
    return {'Priority': str(predictions[0])}
