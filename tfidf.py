import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn as sk
import math
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the email dataset
data = pd.read_csv('dataset.csv')

# Create a new column called "flag"
data['flag'] = data['Label'].map({'spam': 1, 'ham': 0})

# Extract the body of the emails
body = data['Body']


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the body of the emails
vectorizer.fit(body)

# Transform the body of the emails into TF-IDF vectors
tf_idf_vectors = vectorizer.transform(body)

# Calculate the overall TF-IDF score for each email
overall_tf_idf_scores = tf_idf_vectors.sum(axis=1)


# Add the overall TF-IDF scores to the dataset
data['tfidfscore'] = overall_tf_idf_scores

data.head(40)


# Load the email dataset
df = pd.read_csv('tfidf.csv')

# Define a function to assign priority values
def assign_priority(label, tfidfscore):
    if label == 'spam':
        return 'low'
    elif tfidfscore > 3:
        return 'high'
    else:
        return 'medium'

# Apply the function to create a new 'priority' column
df['priority'] = df.apply(lambda row: assign_priority(row['Label'], row['tfidfscore']), axis=1)

df.head(40)
df.to_csv('priority.csv')
