import pandas as pd
import re
import nltk

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
    
    return preprocessed_text


# Load the training dataset
train_file = r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\training.1600000.processed.noemoticon.csv'
train_df = pd.read_csv(train_file, encoding='ISO-8859-1', header=None, names=['target', 'id', 'date', 'query', 'user', 'text'])

# Load the test dataset
test_file = r'C:\Users\ghana\Desktop\sentiment\SentimentAnalysisData\testdata.manual.2009.06.14.csv'
test_df = pd.read_csv(test_file, header=None, names=['target', 'id', 'date', 'query', 'user', 'text'])

# Apply preprocessing to the text columns in the training dataset
train_df['preprocessed_text'] = train_df['text'].apply(preprocess_text)

# Apply preprocessing to the text columns in the test dataset
test_df['preprocessed_text'] = test_df['text'].apply(preprocess_text)

# Save the preprocessed data to separate files
train_df.to_csv('preprocessed_training_data.csv', index=False)
test_df.to_csv('preprocessed_test_data.csv', index=False)

print("Preprocessed Training Data saved to 'preprocessed_training_data.csv'")
print("Preprocessed Test Data saved to 'preprocessed_test_data.csv'")