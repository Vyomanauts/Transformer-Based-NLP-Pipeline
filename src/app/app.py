from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load the trained LSTM model for sentiment classification
lstm_model = tf.keras.models.load_model('lstm_sentiment_model.h5')

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Download stopwords if not already installed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize Keras Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

max_words = 5000  # Limit vocabulary size to 5000 words
keras_tokenizer = Tokenizer(num_words=max_words)
# Fit the tokenizer on your dataset here if not done already.
# keras_tokenizer.fit_on_texts(your_texts)

# Define a function to clean and preprocess text
def preprocess_text(text):
    # Clean the text
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = text.strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Sentiment classification function
def classify_sentiment(text):
    # Preprocess text (same as done during LSTM training)
    processed_text = preprocess_text(text)
    input_data = keras_tokenizer.texts_to_sequences([processed_text])
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=100)  # Match with your LSTM's padding
    prediction = lstm_model.predict(input_data)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment

# Text generation function (GPT-2)
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    sentiment = classify_sentiment(text)
    return jsonify({'sentiment': sentiment})

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generated_text = generate_text(prompt)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
