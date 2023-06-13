import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the vectorized data
train_vectors = sparse.load_npz(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\train_vectors.npz')
test_vectors = sparse.load_npz(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\test_vectors.npz')

# Load the target labels
train_df = pd.read_csv(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\preprocessed_training_data.csv')
test_df = pd.read_csv(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\preprocessed_test_data.csv')
y_train = train_df['target']
y_test = test_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_vectors, y_train, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Function to preprocess text
def preprocess_text(text):
    # Handle NaN values and convert to empty string
    if pd.isnull(text):
        text = ''
    
    # Perform any necessary preprocessing steps
    preprocessed_text = text
    return preprocessed_text

# Function to predict sentiment given a text input
def predict_sentiment(text):
    # Preprocess the input text and convert it to a vector
    preprocessed_text = preprocess_text(text)
    vector = vectorizer.transform([preprocessed_text])
    
    # Make the prediction using the trained model
    sentiment = model.predict(vector)[0]
    
    return sentiment

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training data
train_texts = train_df['preprocessed_text'].fillna('')  # Handle NaN values
vectorizer.fit(train_texts)

# Calculate the accuracy of the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

input_text = "shit it's awesome"
predicted_sentiment = predict_sentiment(input_text)
print("Predicted Sentiment:", predicted_sentiment)
