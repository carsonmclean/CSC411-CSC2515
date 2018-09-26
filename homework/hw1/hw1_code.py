
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.utils import shuffle

# 0 = assignment submission, 1 = regular development, 2 = debugging
PRINT_LEVEL = 1
np.random.seed(12345)

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

    # ekad, tj89, joris -> https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # df = shuffle(df)
    df = df.iloc[np.random.permutation(len(df))]
    pr(2, "data.head()" + str(df.head()))

    vectorizer = TfidfVectorizer()
    # vectorizer = CountVectorizer()
    # as per TA office hour 2018-09-25, fit on all data, not just training.
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

    compute_information_gain(train, 'trumps')

    return datasets, vectorizer

def select_model(datasets, vectorizer):
    train, validation, test = datasets

    # max_depth_list = [i for i in range(1, 30, 1)]
    max_depth_list = [1, 4, 8, 12, 16]
    split_criteria_list = ['entropy', 'gini']

    vocab = ['']*int(len(vectorizer.vocabulary_))
    pr(2, type(vectorizer.vocabulary_))
    for word, index in vectorizer.vocabulary_.items():
        pr(2, (word, index))
        vocab[int(index)] = word
    pr(2, vocab)

    best = [0, "", 0.0] # [depth, criteria, accuracy]
    entropy_accuracy = []
    gini_accuracy = []

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


            correct = 0
            validation_list = validation["label"].tolist()
            for i in range(len(predictions)):
                if predictions[i] == validation_list[i]:
                    correct += 1
            accuracy = (correct/len(predictions)) # or just use sklearn.metrics.accuracy_score(predictions, validation["label"]

            print("\nMax Depth: " + str(max_depth) + " // Split Criteria: " + str(split_criteria))
            print(accuracy)

            if accuracy > best[2]:
                best = [str(max_depth), split_criteria, accuracy]

                # http://scikit-learn.org/stable/modules/tree.html#classification
                graph_data = export_graphviz(
                    decision_tree=classifier,
                    out_file="best.dot",
                    feature_names=vocab,
                    class_names=classifier.classes_,
                    # max_depth=2,
                    filled=True,
                    rounded=True,
                )

            if split_criteria == "entropy":
                entropy_accuracy.append(accuracy)
            else:
                gini_accuracy.append(accuracy)


    print("\nBest Results:")
    print("Max Depth: " + str(best[0]) + " // Split Criteria: " + str(best[1]))
    print(best[2])

    plt.plot(gini_accuracy, label="gini")
    plt.plot(entropy_accuracy, label="entropy")
    plt.legend()
    plt.show()

def compute_information_gain(Y, xi):
    print("COMPUTE_INFORMATION_GAIN")
    print('xi = ' + str(xi))

    real = Y[Y['label'] == 'real'].count()['label']
    fake = Y[Y['label'] == 'fake'].count()['label']
    total = real + fake
    print(real, fake, total)

    print("Parent Entropy:")
    parent_entropy = - (real/total*math.log2(real/total)) - (fake/total*math.log2(fake/total))
    print(parent_entropy)

    print("Splitting on: " + str(xi))
    # df[df['A'].str.contains("hello")]
    subset = Y[Y['titles'].str.contains(xi)]
    real = subset[subset['label'] == 'real'].count()['label']
    fake = subset[subset['label'] == 'fake'].count()['label']
    subset_total = real + fake
    print(real, fake, subset_total)
    ch_entropy = - (real/subset_total*math.log2(real/subset_total)) - (fake/subset_total*math.log2(fake/subset_total))
    print("Child Entropy: " + str(ch_entropy))

    IG = parent_entropy - ch_entropy
    print("Information Gain: " + str(IG))

def pr(level, input):
    if PRINT_LEVEL >= level:
        print(input)


datasets, vectorizer = load_data()
select_model(datasets, vectorizer)