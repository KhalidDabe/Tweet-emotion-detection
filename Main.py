import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from tashaphyne.stemming import ArabicLightStemmer

print("\n")
# raw tweet input
tweet = input("Please enter tweet to classify (type -1 to exit) : ")

while tweet != "-1":

    # instantiate the stemmer
    ArListem = ArabicLightStemmer()

    # this stemmer is used for preceding-waw removal
    isri = ISRIStemmer()

    # read the raw (i.e., unprocessed) tweets into a list
    raw_tweet = [tweet]

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


    # function to remove stopwords if found (parameters are text which is the word token and its compared with stopwords
    # which is a list used later)
    def removeStopwords(text, stopwords):
        for word in text:
            if word in stopwords:
                text.remove(word)


    # function to save emojis if found (parameters are text which is the word token and its compared with stopwords which
    # is a list used later)

    def detectEmojis(text, list):
        output = []
        for word in text:
            if word in list:
                output.append(word)
        return output


    # list to save preprocessed tweet
    preprocessed_tweet = []

    for tweet in raw_tweet:

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
        # add emojis to tweet
        tweetTokens.extend(emojis)
        preprocessed_tweet.append(' '.join(map(str, tweetTokens)))

    # load tfidf file to use it to transform the preprocessed tweet
    # info tfidf values so the model can understand and decide whether positive or negative
    tfidf = pickle.load(open("TFIDF", 'rb'))
    test2 = tfidf.transform(preprocessed_tweet)

    # load random forest model
    rf = pickle.load(open("RF_model", 'rb'))
    random_forest = rf.predict(test2)

    # load Naive Bayes model
    nb = pickle.load(open("NB_model", 'rb'))
    naive_bayes = nb.predict(test2)

    # load Support Vector Machine (SVM) model
    sv = pickle.load(open("SVM_model", 'rb'))
    supp_vec = sv.predict(test2)
    ########################################################Console#################################################################
    print("\n")

    if random_forest == "pos":
        print("Result for Random Forest Model = Positive Tweet")
    else:
        print("Result for Random Forest Model = Negative Tweet")

    print("\n###############################################\n")

    if naive_bayes == "pos":
        print("Result for Naive Bayes model = Positive Tweet")
    else:
        print("Result for Naive Bayes model = Negative Tweet")

    print("\n###############################################\n")

    if supp_vec == "pos":
        print("Result for SVM model = Positive Tweet")
    else:
        print("Result for SVM model = Negative Tweet")

    print("\n###############################################\n")
    print("Preprocessed tweet ===> ", preprocessed_tweet)
    print("\n###############################################\n")
    print("tweet tokens ===> ", tweetTokens)
    print("\n###############################################\n")

    tweet = input("\nPlease enter tweet to classify (type -1 to exit) : ")
