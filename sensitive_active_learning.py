import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from sensitive_labeler import SensitiveLabeler
import pickle as pk

# libact classes
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SVM, Perceptron
from libact.query_strategies import UncertaintySampling, RandomSampling
from sklearn.feature_extraction.text import HashingVectorizer
from os import listdir
from os.path import isfile, join, sep
import pickle


def compare_svm_logistic():
    quota = 40
    n_classes = 2
    E_out1, E_out2 = [], []

    trn_ds, tst_ds, paragraphs = make_dataset_supervised()
    trn_ds2 = copy.deepcopy(trn_ds)

    random_sampling = RandomSampling(trn_ds)

    logRegModel = LogisticRegression()
    svmModel = SVM()

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Error')

    logRegModel.train(trn_ds)
    E_out1 = np.append(E_out1, 1 - logRegModel.score(tst_ds))
    svmModel.train(trn_ds2)
    E_out2 = np.append(E_out2, 1 - svmModel.score(tst_ds))

    query_num = np.arange(0, 1)
    p1, = ax.plot(query_num, E_out1, 'b', label='Log Regr')
    p2, = ax.plot(query_num, E_out2, 'r', label='SVM')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
               shadow=True, ncol=5)
    plt.show(block=False)

    img_ax = fig.add_subplot(2, 1, 2)
    box = img_ax.get_position()
    img_ax.set_position([box.x0, box.y0 - box.height * 0.1, box.width,
                         box.height * 0.9])
    # Give each label its name (labels are from 0 to n_classes-1)
    lbr = SensitiveLabeler(label_name=[str(lbl) for lbl in range(n_classes)],
                           paragraphs=paragraphs)

    labels = []
    # Polls the data for random samples until we have at least one of each label
    while len(labels) < 2:
        ask_id = random_sampling.make_query()
        lb = lbr.label(paragraphs[ask_id], ask_id)
        if lb == -1:
            print("Invalid label. Shutting down")
            return
        if lb not in labels:
            labels.append(lb)
        trn_ds.update(ask_id, lb)
        trn_ds2.update(ask_id, lb)

    # Once we have a sample of each label, we start running uncertainty sampling
    uncertainty_log = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    uncertainty_svm = UncertaintySampling(trn_ds2, method='lc', model=SVM())
    logRegModel.train(trn_ds)
    svmModel.train(trn_ds2)

    for i in range(quota):
        ask_id = uncertainty_log.make_query()
        lb = lbr.label(trn_ds.data[ask_id], ask_id)
        trn_ds.update(ask_id, lb)
        logRegModel.train(trn_ds)

        ask_id = uncertainty_svm.make_query()
        lb = lbr.label(trn_ds2.data[ask_id], ask_id)
        trn_ds2.update(ask_id, lb)
        svmModel.train(trn_ds2)

        E_out1 = np.append(E_out1, 1 - logRegModel.score(tst_ds))
        E_out2 = np.append(E_out2, 1 - svmModel.score(tst_ds))

        ax.set_xlim((0, i + 1))
        ax.set_ylim((0, max(max(E_out1), max(E_out2)) + 0.2))
        query_num = np.arange(0, i + 2)
        p1.set_xdata(query_num)
        p1.set_ydata(E_out1)
        p2.set_xdata(query_num)
        p2.set_ydata(E_out2)
        plt.draw()



def get_paragraphs(filenames):
    """Get the paragraphs from a list of files"""
    # I create a list for the file contents to go in
    import re
    docs = []
    # Then for each file I split it into paragraphs (which is a \n in these
    #   files) and then stick each paragraph into the docs list
    for f in filenames:
        with open(f, 'r') as doc:
            # I use another list comprehension, ensuring that the paragraph is
            #   only added if it's actually a paragraph (isn't 0 or 1 characters
            #   long)


            paragraphs = [paragraph for paragraph in doc.read().split('\n')
                          if len(paragraph.strip()) > 50]
            # extend adds the elements of one list into another list
            docs.extend(paragraphs)

    file_name = "paragraphs"
    file_object = open(file_name, 'wb')
    pk._dump(docs, file_object)
    file_object.close()
    return docs


def build_test(paragraphs):
    file_name = 'test'
    test = []
    if isfile(file_name):
        with open(file_name, 'rb') as f:
            while True:
                try:
                    curr = pickle.load(f)
                except EOFError:
                    break
                else:
                    test.extend(curr)

    file_object = open(file_name, 'wb')

    for i in range(len(test), len(test) + 100):
        if "born" not in paragraphs[i].lower():
            print(paragraphs[i])
            test.append(int(input("0 or 1?")))
        else:
            test.append(1)

    pk._dump(test, file_object)
    file_object.close()


def make_dataset():
    # Change this!
    filenames = ['documents' + sep + f for f in listdir('documents')
                 if isfile(join('documents', f))]
    topics = []
    file_name = 'paragraphs'
    with open(file_name, 'rb') as f:
        while True:
            try:
                topic = pickle.load(f)
            except EOFError:
                break
            else:
                topics.extend(topic)

    paragraphs = np.array(topics)

    # The larger n_features is, the less # of collisions we'll have, but there's a tradeoff. Start large and decrease or vice-versa 2^18
    # preprocessing. Look at every token, classifying them as 'possible year', so that our classifier can look for patterns that include it
    # store the labels after labeling them, so that we don't have to click every time.
    tf_vectorizer = HashingVectorizer(n_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(paragraphs)
    X = tf
    y = [None] * X.shape[0]
    ds = Dataset(X.toarray(), y)

    return tf, ds, paragraphs


def make_lda_dataset():
    # Change this!
    # filenames = ['documents' + sep + f for f in listdir('documents')
    #              if isfile(join('documents', f))]
    # paragraphs = np.array(get_paragraphs(filenames))
    #
    # tf_vectorizer = HashingVectorizer(n_features=100, stop_words='english')
    topics = []
    file_name = 'topics'
    # file_object = open(file_name, 'rb')
    # tf = tf_vectorizer.fit_transform(paragraphs)
    with open(file_name, 'rb') as f:
        while True:
            try:
                topic = pickle.load(f)
            except EOFError:
                break
            else:
                topics.append(topic)

    X = np.array(topics)
    y = [None] * X.shape[0]
    ds = Dataset(X, y)

    attributes = []
    attributes_file_name = 'attributes'
    # file_object = open(file_name, 'rb')
    # tf = tf_vectorizer.fit_transform(paragraphs)
    with open(attributes_file_name, 'rb') as f:
        while True:
            try:
                attribute = pickle.load(f)
            except EOFError:
                break
            else:
                attributes.append(attribute)

    return ds


def make_dataset_supervised():
    n_classes = 2
    classif = []
    class_file_name = 'test'
    file_name = 'paragraphs'
    with open(class_file_name, 'rb') as f:
        while True:
            try:
                topic = pickle.load(f)
            except EOFError:
                break
            else:
                classif.extend(topic)

    y = np.array(classif)

    topics = []
    file_name = 'paragraphs'
    with open(file_name, 'rb') as f:
        while True:
            try:
                topic = pickle.load(f)
            except EOFError:
                break
            else:
                topics.extend(topic)

    X = np.array(topics[:len(y)])

    tf_vectorizer = HashingVectorizer(n_features=100, stop_words='english')
    X = tf_vectorizer.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    while len(np.unique(y_train[:n_classes])) < n_classes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)

    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_classes], [None] * (len(y_train) - n_classes)]))
    tst_ds = Dataset(X_test, y_test)

    return trn_ds, tst_ds, topics[:len(y)]


def main():
    # testing_main()
    quota = 40
    n_classes = 2
    E_out1 = []

    tf, trn_ds, paragraphs = make_dataset()
    trn_ds2 = copy.deepcopy(trn_ds)

    qs2 = RandomSampling(trn_ds2)

    logRegModel = LogisticRegression()

    # fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1)
    # ax.set_xlabel('Number of Queries')
    # ax.set_ylabel('Error')
    #
    # query_num = np.arange(0, 1)
    # p1, = ax.plot(query_num, E_out1, 'b', label='Log Regr')
    #
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
    #            shadow=True, ncol=5)
    # plt.show(block=False)
    #
    # img_ax = fig.add_subplot(2, 1, 2)
    # box = img_ax.get_position()
    # img_ax.set_position([box.x0, box.y0 - box.height * 0.1, box.width,
    #                      box.height * 0.9])

    lbr = SensitiveLabeler(label_name=[str(lbl) for lbl in range(n_classes)],
                           paragraphs=paragraphs)

    labels = []
    while len(labels) < 2:
        ask_id = qs2.make_query()
        lb = lbr.label(trn_ds2.data[ask_id], ask_id)
        if lb == -1:
            print("Invalid label. Shutting down")
            return
        if lb not in labels:
            labels.append(lb)
        trn_ds2.update(ask_id, lb)

    qs = UncertaintySampling(trn_ds2, method='lc', model=LogisticRegression())
    logRegModel.train(trn_ds2)

    for i in range(quota):
        ask_id = qs.make_query()

        lb = lbr.label(trn_ds2.data[ask_id], ask_id)
        trn_ds2.update(ask_id, lb)
        logRegModel.train(trn_ds2)

        # E_out1 = np.append(E_out1, 1 - logRegModel.score(tst_ds))
        # ax.set_xlim((0, i + 1))
        # ax.set_ylim((0, max(E_out1) + 0.2))
        # query_num = np.arange(0, i + 2)
        # p1.set_xdata(query_num)
        # p1.set_ydata(E_out1)

        # plt.draw()

    input("Press any key to continue...")


if __name__ == '__main__':
    main()
