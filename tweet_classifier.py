# imports
import time
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# database communication
db = MongoClient('mongodb://143.215.138.132:27017')['big_data']

# variables
hashtags = ['Trump', 'art'] # choose popular hashtags for enough dataset size
training_set_size = 10000
test_set_size = 1000
smoother = 100