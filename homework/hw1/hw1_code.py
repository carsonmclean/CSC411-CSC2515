from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    fake_file = open("clean_fake.txt", "r")
    real_file = open("clean_real.txt", "r")

    fake_data = fake_file.read().splitlines()
    real_data = real_file.read().splitlines()

    print(fake_data)

    vectorizer = TfidfVectorizer()
    fake_vecs = vectorizer.fit_transform(fake_data)
    real_vecs = vectorizer.fit_transform(real_data)

    print(fake_vecs.shape)
    print(real_vecs.shape)

load_data()