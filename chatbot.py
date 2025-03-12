from flask import Flask, render_template, request, jsonify
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import random
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load fine-tuned GPT-4 model and tokenizer
model = TFGPT2LMHeadModel.from_pretrained("fine-tuned-gpt-4")
tokenizer = GPT2Tokenizer.from_pretrained("fine-tuned-gpt-4")

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
    
    # Generate response using fine-tuned GPT-4 model
    inputs = tokenizer.encode(processed_input, return_tensors="tf")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    app.run(debug=True)