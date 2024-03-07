import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# nltk.download('stopwords')
print(stopwords.words('english')[0])

# Read the CSV file
csv_file_path = "./data/CISC 351 Airbnb Data/TO Detailed Reviews CISC 351.csv"
df = pd.read_csv(csv_file_path)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Tokenize reviews and remove stop words
stop_words = set(stopwords.words('english'))

# Function to tokenize a review and remove stop words
def tokenize_and_remove_stopwords(review):
    tokens = word_tokenize(review.lower())
    return [token for token in tokens if token.isalpha() and token not in stop_words]

# Tokenize and remove stop words for each review
df['tokens'] = df['comments'].apply(tokenize_and_remove_stopwords)

# Apply sentiment analysis to each review
df['sentiment'] = df['comments'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Define a function to count the occurrences of words in positive and negative reviews
def count_word_sentiment(word, sentiment):
    return sum(1 for tokens, s in zip(df['tokens'], df['sentiment']) if word in tokens and s * sentiment > 0)

# Create a list of unique words
all_words = set(word for tokens in df['tokens'] for word in tokens)

# Calculate the correlation between each word and sentiment
correlation_matrix = pd.DataFrame(index=all_words, columns=['Positive', 'Negative'])

for word in all_words:
    positive_count = count_word_sentiment(word, 1)
    negative_count = count_word_sentiment(word, -1)
    correlation_matrix.loc[word] = [positive_count, negative_count]

# Normalize the counts by dividing by the total number of positive and negative reviews
correlation_matrix['Positive'] /= sum(df['sentiment'] > 0)
correlation_matrix['Negative'] /= sum(df['sentiment'] < 0)

# Output the correlation matrix
print(correlation_matrix)
