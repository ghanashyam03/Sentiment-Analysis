import sys
import pandas as pd
import re
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Get the set of stopwords
stopwords_set = set(stopwords.words('english'))

# Text cleaning and preprocessing using NLTK
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize the text
        tokens = text.split()

        # Lemmatization and remove stopwords
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]

        # Join tokens back to a single string
        preprocessed_text = ' '.join(tokens)
    else:
        preprocessed_text = ''  # Return empty string for np.nan values

    return preprocessed_text


# Load the preprocessed data
train_df = pd.read_csv(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\preprocessed_training_data.csv')
test_df = pd.read_csv(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\preprocessed_test_data.csv')

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Preprocess the training data
preprocessed_train_text = train_df['preprocessed_text'].apply(preprocess_text)

# Fit the vectorizer on the preprocessed training data
vectorizer.fit(preprocessed_train_text)

# Preprocess the test data
preprocessed_test_text = test_df['preprocessed_text'].apply(preprocess_text)

# Vectorize the preprocessed test data
test_vectors = vectorizer.transform(preprocessed_test_text)

# Load the trained model
model = MultinomialNB()
model.fit(vectorizer.transform(preprocessed_train_text), train_df['target'])

# Predict the sentiment for test data
predicted_sentiments = model.predict(test_vectors)

# Get the true sentiments from test data
true_sentiments = test_df['target']

# Build the confusion matrix
cm = confusion_matrix(true_sentiments, predicted_sentiments)

# Print the confusion matrix
print(cm)
