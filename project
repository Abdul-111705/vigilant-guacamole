import tweepy

# Add your Twitter API credentials
consumer_key = '9b7zFQJ12VoaMFZP39kVSTy3q'
consumer_secret = 'RXmiRLegJqzmW8FFHkRLQFLOFZeyL1EuAVThlOpzOFUT0ZYHQ1'
access_token = '1707741658756837376-KxlAdicBbJtN43dIsRHOsfsIayJhDT'
access_token_secret = 'ihaNrWY3DFMGg6ndHgIHGHzJarY0rj6OZEvaPCZT78rxk'

# Authentication with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Collect tweets for a specific stock
def collect_tweets(stock_symbol, num_tweets):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=stock_symbol, lang='en').items(num_tweets):
        tweets.append(tweet.text)
    return tweets

# Example usage
stock_symbol = 'AAPL'
num_tweets = 1000
tweets = collect_tweets(stock_symbol, num_tweets)

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (one-time step)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocess tweet text
def preprocess_tweet(tweet):
    # Remove URLs, special characters, and numbers
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = re.sub(r'\d+', '', tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenization and stopword removal
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(tweet)
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# Preprocess all collected tweets
preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis
sentiment_scores = [sid.polarity_scores(tweet) for tweet in preprocessed_tweets]

# Create a DataFrame for the sentiment scores
df = pd.DataFrame(sentiment_scores)

# Classify each tweet as positive, negative, or neutral
def classify_sentiment(compound):
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['compound'].apply(classify_sentiment)

# Display the DataFrame with sentiment analysis results
print(df)

import matplotlib.pyplot as plt

# Plot the sentiment distribution
sentiment_distribution = df['sentiment'].value_counts()
plt.bar(sentiment_distribution.index, sentiment_distribution.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis of Tweets')
plt.show()
