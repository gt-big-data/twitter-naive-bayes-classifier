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

match = {'$match': {'timestamp': {'$gte': time.time() - 2 * 7 * 24 * 3600}, 'hashtags': {'$in': hashtags}}}
sample = {'$sample': {'size': training_set_size + test_set_size}}
pipeline = [match, sample]
query_result = db.tweet.aggregate(pipeline)

# preprocessing all data
hashtag_list = []
word_list = []
text_list = []
for tweet in query_result:
    hashtag_list.append(set(tweet['hashtags']).intersection(set(hashtags)).pop())
    word_list.append(' '.join(tweet['words']))
    text_list.append(tweet['text'])

# counting
word_counts = CountVectorizer().fit_transform(word_list)

# split training set and test set
training_hashtag_list = hashtag_list[:training_set_size]
test_hashtag_list = hashtag_list[training_set_size:]
training_text_list = text_list[:training_set_size]
test_text_list = text_list[training_set_size:]
training_word_counts = word_counts[:training_set_size]
test_word_counts = word_counts[:training_set_size]

# naive bayes
clf = MultinomialNB(alpha=smoother).fit(training_word_counts, training_hashtag_list)

# prediction
predicted = clf.predict(test_word_counts)

# printing texts and predicted hashtags
for text, predicted_hashtag in zip(test_text_list, predicted):
    print('Text: %s' % text)
    print('Predicted Hashtag: #%s\n' % predicted_hashtag)

# print accuracy
print('Accuracy: %.1f%%' % (sum([float(hashtag == predicted_hashtag) for hashtag, predicted_hashtag
    in zip(test_hashtag_list, predicted)]) / test_set_size * 100))
