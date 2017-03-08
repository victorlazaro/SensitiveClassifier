import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from sensitive_labeler import SensitiveLabeler
import pickle as pk

# libact classes
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SVM
from libact.query_strategies import UncertaintySampling, RandomSampling
from sklearn.feature_extraction.text import HashingVectorizer
from os import listdir
from os.path import isfile, join, sep

n_topics = 100
n_features = 1000

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

    # file_name = "paragraphs"
    # file_object = open(file_name, 'wb')
    # pk._dump(docs, file_object)
    # file_object.close()
    return docs_non, docs_sens


def build_test(paragraphs):
    file_name = 'test'
    test = []
    if isfile(file_name):
        with open(file_name, 'rb') as f:
            while True:
                try:
                    curr = pk.load(f)
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
    n_classes = 2
    n_labeled = 10
    filenames = ['documents' + sep + f for f in listdir('documents')
                 if isfile(join('documents', f))]

    docs_non, docs_sen = get_paragraphs()
    mixed = np.append(docs_sen, docs_non)

    docs_non = np.array(docs_non)
    docs_sen = np.array(docs_sen)
    labels = [0] * len(mixed)
    for i in range(len(docs_sen)):
        labels[i] = 1
    # paragraphs = np.array(get_paragraphs(filenames))



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
    y = [None] * X.shape[0]
    ds = Dataset(X, y)


    # X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=12, test_size=0.33)
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(mixed, labels, random_state=12, test_size=0.33)
    #
    #
    #
    # trn_ds = Dataset(X_train, np.concatenate(
    #     [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    # tst_ds = Dataset(X_test, y_test)
    #
    # trn_ds2 = Dataset(X_train2, np.concatenate(
    #     [y_train2[:n_labeled], [None] * (len(y_train2) - n_labeled)]))
    #
    #
    # return tf, trn_ds, tst_ds, mixed, trn_ds2
    return tf, ds, mixed





def main():
    quota = 40
    n_classes = 2
    E_out1 = []

    # tf, trn_ds, tst_ds, paragraphs, trn_ds_paragraphs = make_dataset()

    tf, trn_ds, paragraphs = make_dataset()
    trn_ds = copy.deepcopy(trn_ds)

    random_sampling = RandomSampling(trn_ds)

    logRegModel = LogisticRegression()

    # fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1)
    # ax.set_xlabel('Number of Queries')
    # ax.set_ylabel('Error')
    #
    # logRegModel.train(trn_ds)
    # E_out1 = np.append(E_out1, 1 - logRegModel.score(tst_ds))
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
        ask_id = random_sampling.make_query()
        # lb = lbr.label(trn_ds_paragraphs.data[ask_id], ask_id)
        lb = lbr.label(paragraphs[ask_id], ask_id)
        if lb == -1:
            print("Invalid label. Shutting down")
            return
        if lb not in labels:
            labels.append(lb)
        trn_ds.update(ask_id, lb)

    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    logRegModel.train(trn_ds)

    for i in range(quota):
        ask_id = qs.make_query()

        lb = lbr.label(paragraphs[ask_id], ask_id)
        # lb = lbr.label(trn_ds_paragraphs.data[ask_id], ask_id)
        trn_ds.update(ask_id, lb)
        logRegModel.train(trn_ds)

        # E_out1 = np.append(E_out1, 1 - logRegModel.score(tst_ds))
        # ax.set_xlim((0, i + 1))
        # ax.set_ylim((0, max(E_out1) + 0.2))
        # query_num = np.arange(0, i + 2)
        # p1.set_xdata(query_num)
        # p1.set_ydata(E_out1)
        #
        # plt.draw()

    input("Press any key to continue...")


if __name__ == '__main__':
    main()
