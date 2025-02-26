import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download("punkt_tab")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load training data
with open("data/intents.json") as file:
    intents = json.load(file)

# Data preprocessing
patterns = []
labels = []
responses = {}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        patterns.append(" ".join([lemmatizer.lemmatize(w.lower()) for w in word_list]))
        labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
y = labels

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
pickle.dump({"model": model, "vectorizer": vectorizer, "responses": responses}, open("models/chatbot_model.pkl", "wb"))

print("Chatbot trained successfully!")
