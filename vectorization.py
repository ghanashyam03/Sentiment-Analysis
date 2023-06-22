import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the preprocessed data
train_df = pd.read_csv(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\preprocessed_training_data.csv')
test_df = pd.read_csv(r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\preprocessed_test_data.csv')

# Replace NaN values with empty strings
train_df['preprocessed_text'] = train_df['preprocessed_text'].fillna('')
test_df['preprocessed_text'] = test_df['preprocessed_text'].fillna('')

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
train_vectors = vectorizer.fit_transform(train_df['preprocessed_text'])

# Transform the test data
test_vectors = vectorizer.transform(test_df['preprocessed_text'])

# Save the vectorized data as sparse matrices
train_vector_file = r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\train_vectors.npz'
test_vector_file = r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\test_vectors.npz'
sparse.save_npz(train_vector_file, train_vectors)
sparse.save_npz(test_vector_file, test_vectors)

print("Vectorization complete.")

