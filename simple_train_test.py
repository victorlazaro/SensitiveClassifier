import numpy as np
from libact.base.dataset import Dataset
from os.path import isfile

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

n_topics = 100
n_features = 1000

def preprocessing(paragraphs):
    import re
    processed = []
    for paragraph in paragraphs:
        processed.append(re.sub(r'(?<![0-9])((([0-2][0-9][0-9][0-9])|([0-9][0-9][0-9])|([0-9][0-9])|([0-9])))(?![0-9])', "possibleYear", paragraph))

    return processed


def get_paragraphs():
    """Get the paragraphs from a list of files"""
    # I create a list for the file contents to go in
    docs_non = []
    docs_sens = []
    non_sensitive_file = "test-non"
    sensitive_file = "test-sensitive"

    if (isfile(non_sensitive_file)):
        with open(non_sensitive_file, 'r') as doc:
            paragraphs = [paragraph for paragraph in doc.read().split('\n')]
            docs_non.extend(paragraphs)

    if (isfile(sensitive_file)):
        with open(sensitive_file, 'r') as doc:
            paragraphs = [paragraph for paragraph in doc.read().split('\n')]
            docs_sens.extend(paragraphs)



    docs_non = preprocessing(docs_non)
    docs_sens = preprocessing(docs_sens)


    return docs_non, docs_sens


def make_dataset():
    # Change this!
    n_classes = 2
    n_labeled = 10

    docs_non, docs_sen = get_paragraphs()
    mixed = np.append(docs_sen, docs_non)

    docs_non = np.array(docs_non)
    docs_sen = np.array(docs_sen)
    labels = [0] * len(mixed)
    for i in range(len(docs_sen)):
        labels[i] = 1



    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                        stop_words='english')

    tf = tf_vectorizer.fit_transform(mixed)
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    doc_topic_matrix = lda.transform(tf)
    X = doc_topic_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=12, test_size=0.33)

    return X_train, X_test, y_train, y_test


def main():

    X_train, X_test, y_train, y_test = make_dataset()

    logRegModel = LogisticRegression()

    # new_x = logRegModel.fit_transform(X_train, y_train)
    prediction = logRegModel.predict(X_test)

    print(accuracy_score(prediction, y_test))
    # print(logRegModel.score(prediction, y_test))


    input("Press any key to continue...")


if __name__ == '__main__':
    main()
