from flask import Flask, render_template, request, jsonify
import pickle
import nltk
import random
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load trained chatbot model
data = pickle.load(open("models/chatbot_model.pkl", "rb"))
model = data["model"]
vectorizer = data["vectorizer"]
responses = data["responses"]

lemmatizer = WordNetLemmatizer()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    
    # Preprocess user input
    words = nltk.word_tokenize(user_message)
    processed_input = " ".join([lemmatizer.lemmatize(w.lower()) for w in words])
    
    # Convert input to model format
    X_input = vectorizer.transform([processed_input])
    
    # Predict intent tag
    predicted_tag = model.predict(X_input)[0]
    
    # Get bot response
    bot_response = random.choice(responses[predicted_tag])

    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
