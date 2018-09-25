
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

import numpy as np
import pandas
import math

# 0 = assignment submission, 1 = regular development, 2 = debugging
PRINT_LEVEL = 1


def load_data():
    fake_file = open("clean_fake.txt", "r")
    real_file = open("clean_real.txt", "r")
    fake_data = fake_file.read().splitlines()
    real_data = real_file.read().splitlines()


    fake_df = pandas.DataFrame({'titles': fake_data, 'label': "fake"})
    real_df = pandas.DataFrame({'titles': real_data, 'label': "real"})
    df = fake_df.append(real_df)
    print("fake.shape()" + str(fake_df.shape))
    print("real.shape()" + str(real_df.shape))
    print("data.shape()" + str(df.shape))

    # ekad, tj89 -> https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    df = shuffle(df)
    pr(2, "data.head()" + str(df.head()))

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['titles'])
    pr(2, vectorizer.vocabulary_)

    count = df.shape[0]
    train = df[0:math.floor(count*0.7)]
    validation = df[math.floor(count*0.7):math.floor(count*0.85)]
    test = df[math.floor(count*0.85):]
    print("\ntrain: " + str(train.shape))
    print("validation: " + str(validation.shape))
    print("test: " + str(test.shape))
    print("\n% real labelled data in each set:")
    print(train[train["label"] == "real"].count()["label"]/train.shape[0])
    print(validation[validation["label"] == "real"].count()["label"]/validation.shape[0])
    print(test[test["label"] == "real"].count()["label"]/test.shape[0])
    print("% real labelled data overall:")
    print(len(real_data)/(len(fake_data)+len(real_data)))

    datasets = [train, validation, test]

    return datasets, vectorizer

def select_model(datasets, vectorizer):
    train, validation, test = datasets

    max_depth_list = [1,3,5,10,25]
    split_criteria_list = ['entropy', 'gini']

    for max_depth in max_depth_list:
        for split_criteria in split_criteria_list:
            classifier = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=split_criteria
            )
            classifier.fit(
                X=vectorizer.transform(train["titles"]),
                y=train["label"]
            )
            predictions = classifier.predict(vectorizer.transform(validation["titles"]))
            print("\nMax Depth: " + str(max_depth) + " Split Criteria: " + str(split_criteria))
            print(accuracy_score(predictions, validation["label"]))



def old():
    fake_file = open("clean_fake.txt", "r")
    real_file = open("clean_real.txt", "r")

    fake_data = fake_file.read().splitlines()
    real_data = real_file.read().splitlines()

    titles = []
    labels = []

    for title in fake_data:
        titles.append(title)
        labels.append("fake")

    for title in real_data:
        titles.append(title)
        labels.append("real")

    df = pandas.DataFrame()
    df["titles"] = titles
    df["labels"] = labels

    print(df.head())

    train_titles, test_titles, train_labels, test_labels = train_test_split(df["titles"], df["labels"])


    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["titles"])
    train_vecs = vectorizer.transform(train_titles)
    test_vecs = vectorizer.transform(test_titles)

    classifier = DecisionTreeClassifier()
    classifier.fit(train_vecs, train_labels)
    predictions = classifier.predict(test_vecs)

    print(accuracy_score(predictions, test_labels))

def pr(level, input):
    if PRINT_LEVEL >= level:
        print(input)


datasets, vectorizer = load_data()
select_model(datasets, vectorizer)