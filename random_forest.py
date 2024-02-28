import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# dictionary to load the preprocessed data to vectorize it

# dictionary to load preprocessed tweets from preprocessed tweets file
preprocessed_dict = {}

csv_filename = 'preprocessed_tweets.csv'
with open(csv_filename, encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        preprocessed_dict[row[0]] = row[1]

vectorizer = TfidfVectorizer(token_pattern=r'[^\s]+')

# produce tfidf values
X = vectorizer.fit_transform(preprocessed_dict.keys())

pickle.dump(vectorizer, open("TFIDF", 'wb'))

###########################################################################


# labels of sentences in our corpus
y = list(preprocessed_dict.values())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# n_estimators is the number of trees in the forest, this can be
# a parameter that you can play with to fine tune your model
RForest_clf = RandomForestClassifier(n_estimators=100)

# train the model using 25% of the dataset
classifier = RForest_clf.fit(X_train, y_train)

# test the model using the remaining 75% of the dataset
y_pred = RForest_clf.predict(X_test)

# find the accuracy
print('\n'"Accuracy of our Random Forst Classifier is: ",
      metrics.accuracy_score(y_test, y_pred) * 100, "\n")

print(metrics.classification_report(y_test, y_pred), "\n")

# cross validation for random forest
cvRF = cross_val_score(RForest_clf, X, y)

print("5 - cross validation results for random forest : \n\n")

for i in range(len(cvRF)):
    print("Fold number " + str(i + 1) + " result: " + str(cvRF[i]))

# save model

with open('RF_model', 'wb') as f:
    pickle.dump(classifier, f)

class_names = ["pos", "neg"]

# plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Reds,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    print()

plt.show()
