# Transformer-Based-NLP-Pipeline

# Sentiment Analysis and Text Generation Project

## Overview
This project implements a sentiment analysis model and a text generation model using the IMDB dataset. The primary goal is to classify movie reviews as positive or negative and generate contextually relevant text based on given prompts. The project leverages deep learning techniques, specifically an LSTM-based model for sentiment classification and a fine-tuned GPT-2 model for text generation.

## Dataset
The dataset used for this project is the [IMDB Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), which contains 50,000 movie reviews labeled as positive or negative. 

**Note:** The dataset is not included in this repository. Please download it from the provided link and ensure the CSV file is named `IMDB_Dataset.csv`.

## Assumptions and Decisions

1. **Model Choice**: Initially, the project intended to utilize a pre-trained BERT model for the sentiment analysis task. However, due to integration challenges and performance considerations, a Bidirectional LSTM model was implemented instead. This decision was made to ensure timely project completion while still achieving effective sentiment classification.

2. **Data Quality**: It is assumed that the dataset used is clean and well-structured after the preprocessing steps. Any inherent biases or inconsistencies in the data may affect the model's performance.

3. **Environment Compatibility**: The code and models are assumed to run in a compatible Python environment with all specified dependencies correctly installed.
