import csv
import pandas
import re
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from tashaphyne.stemming import ArabicLightStemmer


# instantiate the stemmer
ArListem = ArabicLightStemmer()

# this stemmer is used for preceding-waw removal
isri = ISRIStemmer()

# read the raw (i.e., unprocessed) tweets into a dictionary, where the key is the tweet and the value is the sentiment
raw_tweet_dictionary = {}
with open("PositiveTweets.tsv", encoding="utf-8") as f:
    for line in f:
        (sentiment, tweet) = line.replace("\n", "").split('\t')
        raw_tweet_dictionary[tweet] = sentiment

for i, tweet in enumerate(raw_tweet_dictionary):
    print(f'{tweet} : {raw_tweet_dictionary[tweet]}')

# emoji file
allEmojis = pandas.read_csv('emojis.csv')
allEmojisList = []
for i in allEmojis[allEmojis.columns[0]]:
    allEmojisList.append(i.split(';')[0])

# stopwords
stopwordsList = []
stopwords = pandas.read_csv('list.txt')
for i in stopwords[stopwords.columns[0]]:
    stopwordsList.append(i)


def removeStopwords(text, stopwords):
    for word in text:
        if word in stopwords:
            text.remove(word)


def detectEmojis(text, list):
    output = []
    for word in text:
        if word in list:
            output.append(word)
    return output


preprocessed_tweet_dict = {}

for tweet in raw_tweet_dictionary:

    # remove underscore in hashtags and replace it with a whitespace, also I've seen some occurrences where "pos" was
    # still being printed so I removed it
    processed_tweet = tweet.replace("_", " ")
    emojis = detectEmojis(tweet, allEmojisList)
    tweetTokens = word_tokenize(re.sub(r'[^\w\s]|\d', '', tweet))
    removeStopwords(tweetTokens, stopwordsList)

    # remove waw al3atf + stem
    for token in tweetTokens:
        w_token = isri.waw(token)
        stem = ArListem.light_stem(w_token)
        tweetTokens.append(ArListem.get_stem())
        tweetTokens.remove(token)

    tweetTokens.extend(emojis)
    preprocessed_tweet_dict[' '.join(map(str, tweetTokens))] = raw_tweet_dictionary[tweet]

# Open the file in writing mode (no blank lines)
with open('preprocessed_tweets.csv', 'a', newline='', encoding="utf-8") as f:
    # Create a CSV writer object
    writer = csv.writer(f)
    # Write one key-value tuple per row
    for row in preprocessed_tweet_dict.items():
        writer.writerow(row)

#############NegativeTweets###############

raw_tweet_dictionary = {}
with open("NegativeTweets.tsv", encoding="utf-8") as f:
    for line in f:
        (sentiment, tweet) = line.replace("\n", "").split('\t')
        raw_tweet_dictionary[tweet] = sentiment

for i, tweet in enumerate(raw_tweet_dictionary):
    print(f'{tweet} : {raw_tweet_dictionary[tweet]}')

tweet_dict = {}

for tweet in raw_tweet_dictionary:

    # remove underscore in hashtags and replace it with a whitespace, also I've seen some occurrences where "pos" was
    # still being printed so I removed it
    processed_tweet = tweet.replace("_", " ")
    emojis = detectEmojis(tweet, allEmojisList)
    tweetTokens = word_tokenize(re.sub(r'[^\w\s]|\d', '', tweet))
    removeStopwords(tweetTokens, stopwordsList)

    # remove waw al3atf + stem
    for token in tweetTokens:
        w_token = isri.waw(token)
        stem = ArListem.light_stem(w_token)
        tweetTokens.append(ArListem.get_stem())
        tweetTokens.remove(token)

    tweetTokens.extend(emojis)
    tweet_dict[' '.join(map(str, tweetTokens))] = raw_tweet_dictionary[tweet]

# Open the file in writing mode (no blank lines)
with open('preprocessed_tweets.csv', 'a', newline='', encoding="utf-8") as f:
    # Create a CSV writer object
    writer = csv.writer(f)
    # Write one key-value tuple per row
    for row in tweet_dict.items():
        writer.writerow(row)
