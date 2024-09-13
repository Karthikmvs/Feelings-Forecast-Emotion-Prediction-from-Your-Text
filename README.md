# Feelings-Forecast-Emotion-Prediction-from-Your-Text
This project is an Emotion Prediction App that uses various machine learning models and a Long Short-Term Memory (LSTM) network to classify emotions from text input. The app is built using Streamlit and provides predictions along with emojis representing the emotions and a confidence level.

## Steps Taken

### 1. Data Preparation

1. **Data Loading and Exploration:**
   - Loaded and explored the dataset containing text comments and corresponding emotion labels.
   - Performed data cleaning by removing duplicates and visualizing the data distribution.

2. **Text Preprocessing:**
   - Implemented text cleaning functions to preprocess the text data.
   - Removed stopwords, applied stemming, and tokenized text using TF-IDF vectorization for traditional machine learning models.

### 2. Model Training

1. **Traditional Machine Learning Models:**
   - Evaluated various models including Naive Bayes, Logistic Regression, Random Forest, SVM, K-Nearest Neighbors, Decision Tree, Gradient Boosting, and AdaBoost.
   - Selected XGBoost as the best model based on performance.

2. **Deep Learning Model:**
   - Created an LSTM model using TensorFlow’s Keras for emotion classification.
   - Preprocessed text data with one-hot encoding and padding sequences.
   - Trained the LSTM model and saved it along with the label encoder and tokenizer.

### 3. Model Performance

#### Accuracy

**1. Multinomial Naive Bayes:** 68.23%  
**2. Logistic Regression:** 84.75%  
**3. Random Forest:** 85.02%  
**4. Support Vector Machine:** 83.38%  
**5. K-Nearest Neighbors:** 76.35%  
**6. Decision Tree:** 80.80%  
**7. Gradient Boosting:** 79.53%  
**8. AdaBoost:** 36.03%  
**9. XGBoost:** 85.50%  
**10. LSTM:** 97.19%

## Model Deployment

The trained models are used in the Streamlit app to provide real-time emotion predictions from user input. The LSTM model is utilized for its high accuracy, while other models are included for comparison and fallback options.

## Future Work

- Enhance the app with more advanced features and better user interface.
- Continue experimenting with different architectures and hyperparameters to further improve model performance.
## Files Included

1. **`app.py`**: Streamlit application script for running the web interface that predicts emotions based on user input.
2. **`Emotion_Detection.ipynb`**: Jupyter Notebook containing code for data preprocessing, model training, and evaluation. Includes comparisons of various classifiers.
3. **`labelencoder.pkl`**: Pickle file with the LabelEncoder object used for encoding and decoding emotion labels.
4. **`lstm_model.h5`**: Saved LSTM model file used for emotion prediction.
5. **`tokenizer.pkl`**: Pickle file with the Tokenizer object used for text tokenization.
6. **`train.txt`**: Training data file containing text samples and corresponding labels.
7. **`vocab_info.pkl`**: Pickle file with vocabulary information used in the training process.
