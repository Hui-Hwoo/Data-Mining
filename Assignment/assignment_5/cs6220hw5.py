# cleaning the texts

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")


def clean_text(training_data, testing_data):

    ps = PorterStemmer()

    def portstem(review):
        return [
            ps.stem(word)
            for word in review
            if not word in set(stopwords.words("english"))
        ]

    new_data = pd.concat([training_data, testing_data])

    processed_reviews = new_data["Review"].apply(
        lambda x: re.sub("[^a-zA-Z]", " ", str(x))
    )  # remove non-alphabets
    
    processed_reviews = processed_reviews.map(
        lambda x: x.lower()
    )  # convert to lower case
    processed_reviews = processed_reviews.map(lambda x: x.split())  # split the words
    processed_reviews = processed_reviews.map(
        portstem
    )  # remove stopwords and stem the words
    processed_reviews = processed_reviews.map(lambda x: " ".join(x))  # join the words

    # new_data["Review"] = processed_reviews
    # training_corpus = new_data[: len(training_data)]
    # testing_corpus = new_data[len(training_data) :]

    training_corpus = processed_reviews[: len(training_data)] # split the training and testing data
    testing_corpus = processed_reviews[len(training_data) :]

    return training_corpus, testing_corpus
