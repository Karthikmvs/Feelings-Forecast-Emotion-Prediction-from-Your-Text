import streamlit as st
import numpy as np
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords  
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Download necessary NLTK data
nltk.download('stopwords')

# Emoji dictionary for the emotions
emotion_emoji = {
    'sadness': '😢',
    'anger': '😠',
    'love': '❤️',
    'surprise': '😮',
    'fear': '😨',
    'joy': '😊'
}

# Sentence cleaning function
def sentence_cleaning(sentence, max_len=300):
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    text = " ".join(text)
    corpus.append(text)

    # Use the same tokenizer used during training
    sequences = tokenizer.texts_to_sequences(corpus)
    pad = pad_sequences(sequences, maxlen=max_len, padding='pre')

    return pad

# Load the model and label encoder
model = load_model('lstm_model.h5')
lb = pickle.load(open('labelencoder.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Streamlit app
st.markdown("<h1 style='color: #FF6F61; text-align: center;'>Emotion Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #7B68EE;'>Predict emotions like: Sadness, Anger, Love, Surprise, Fear, Joy</h3>", unsafe_allow_html=True)
st.write("Enter any text below, and the model will predict the emotion associated with it.")

# Input text field
input_text = st.text_input('Enter your text here:')

# Button for prediction
if st.button('Predict'):
    if input_text.strip() == "":
        st.warning('Please enter some text to analyze.')
    else:
        cleaned_sentence = sentence_cleaning(input_text)
        prediction = model.predict(cleaned_sentence)
        result = lb.inverse_transform(np.argmax(prediction, axis=-1))[0]
        proba = float(np.max(prediction))

        # Display result with emoji
        emoji = emotion_emoji.get(result, '')
        st.markdown(f"<h2 style='color: #32CD32;'>The predicted emotion is: <b>{result} {emoji}</b></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #2F4F4F;'>Prediction confidence: <b>{proba:.2f}</b></p>", unsafe_allow_html=True)

        # Display probability as a progress bar
        st.progress(proba)

# Footer with extra information
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("This app uses an LSTM model trained to classify text into one of six emotions: Sadness 😢, Anger 😠, Love ❤️, Surprise 😮, Fear 😨, and Joy 😊.")
st.markdown("<p style='text-align: center; color: #808080;'>Created by M.V.S.KARTHIK</p>", unsafe_allow_html=True)