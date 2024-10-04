import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already installed
nltk.download('stopwords')

# Load the dataset (ensure the CSV file is in the correct path)
df = pd.read_csv('IMDB_Dataset.csv')

# Define a function to clean the text
def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = text.strip()
    return text

# Apply the cleaning function to the review column
df['cleaned_review'] = df['review'].apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['cleaned_review'] = df['cleaned_review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Tokenize the cleaned reviews using Keras Tokenizer
max_words = 5000  # Limit vocabulary size to 5000 words
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['cleaned_review'])

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])

# Pad sequences to ensure uniform length
max_len = 100  # Define a maximum length for padding
X = pad_sequences(sequences, maxlen=max_len)

# Prepare labels (binary sentiment: positive/negative)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
y = df['sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessing Complete. Data is ready for model training.")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Parameters
embedding_dim = 128
lstm_units = 128
input_length = X_train.shape[1]  # Length of padded sequences

# Build the LSTM model
model = Sequential()

# Embedding layer (input_dim = max_words)
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=input_length))

# Add a bidirectional LSTM layer
model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=False)))

# Add dropout to prevent overfitting
model.add(Dropout(0.5))

# Add a fully connected (Dense) layer
model.add(Dense(128, activation='relu'))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
model.save('lstm_sentiment_model.h5')
