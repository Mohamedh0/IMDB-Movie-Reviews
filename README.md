# IMDB Movie Reviews Sentiment Analysis

This project focuses on sentiment analysis of movie reviews from the IMDB dataset. It uses natural language processing (NLP) techniques and deep learning to classify movie reviews as positive or negative. The workflow includes data preprocessing, exploratory data analysis, and building a neural network model using TensorFlow and Keras.

## Dataset
- **Source:** IMDB Dataset of 50,000 Movie Reviews
- **Columns:** 
  - `review`: Textual movie review.
  - `sentiment`: Sentiment label (`positive` or `negative`).

## Key Steps

### 1. **Libraries Used**
The project leverages several Python libraries for data manipulation, visualization, and modeling:
- Data Analysis: `pandas`, `numpy`
- Text Processing: `nltk`, `re`, `gensim`
- Visualization: `matplotlib`, `seaborn`
- Machine Learning: `scikit-learn`
- Deep Learning: `TensorFlow`, `Keras`

### 2. **Data Preprocessing**
- **Text Cleaning:** Removing HTML tags, special characters, and converting text to lowercase.
- **Tokenization:** Splitting text into words using the Keras tokenizer.
- **Padding:** Standardizing review lengths for neural network input.
- **Embedding:** Using pre-trained FastText embeddings for word representation.

### 3. **Model Building**
A sequential neural network model is built with:
- **Embedding Layer:** Word embeddings from FastText.
- **LSTM Layer:** Captures sequential dependencies in text.
- **Dropout:** Reduces overfitting.
- **Dense Layer:** For final classification.

### 4. **Evaluation**
- Metrics like accuracy and loss.

[Kaggle Notebook]([https://www.kaggle.com/](https://www.kaggle.com/code/hamoi9/imdb-movie-reviews)) 
