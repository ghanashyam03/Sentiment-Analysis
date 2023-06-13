import pandas as pd

# Load the training dataset
train_file = 'C:\\Users\\ghana\\Downloads\\SentimentAnalysisData\\training.1600000.processed.noemoticon.csv'
train_df = pd.read_csv(train_file, header=None, names=['target', 'id', 'date', 'query', 'user', 'text'], encoding='latin1')


# Load the test dataset
test_file = 'C:\\Users\\ghana\\Downloads\\SentimentAnalysisData\\testdata.manual.2009.06.14.csv'
test_df = pd.read_csv(test_file, header=None, names=['target', 'id', 'date', 'query', 'user', 'text'], encoding='latin1')
print("Training data:")
print(train_df.head())

print("\nTest data:")
print(test_df.head())
