import copy
import numpy as np
import matplotlib.pyplot as plt
from Sensitive_Labeler import SensitiveLabeler

# libact classes
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from libact.labelers import InteractiveLabeler
from os import listdir
from os.path import isfile, join, sep

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
                          if len(paragraph.strip()) > 20]
            # extend adds the elements of one list into another list
            docs.extend(paragraphs)
    return docs


def make_dataset():

    # Change this!
    filenames = ['documents' + sep + f for f in listdir('documents')
                 if isfile(join('documents', f))]
    paragraphs = np.array(get_paragraphs(filenames))

    tf_vectorizer = HashingVectorizer(n_features=100,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(paragraphs)
    X = tf
    y = [None] * X.shape[0]
    print(len(y))
    ds = Dataset(X, y)

    return tf, ds, paragraphs


def main():
    quota = 6
    n_classes = 2
    E_out1, E_out2 = [], []

    tf, trn_ds, paragraphs = make_dataset()
    trn_ds2 = copy.deepcopy(trn_ds)


    # qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    qs2 = RandomSampling(trn_ds2)

    model = LogisticRegression()

    # fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1)
    # ax.set_xlabel('Number of Queries')
    # ax.set_ylabel('Error')
    #
    # model.train(trn_ds)
    # E_out1 = np.append(E_out1, 1 - model.score(tst_ds))
    # model.train(trn_ds2)
    # E_out2 = np.append(E_out2, 1 - model.score(tst_ds))

    # query_num = np.arange(0, 1)
    # p1, = ax.plot(query_num, E_out1, 'g', label='qs Eout')
    # p2, = ax.plot(query_num, E_out2, 'k', label='random Eout')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
    #            shadow=True, ncol=5)
    # plt.show(block=False)
    #
    # img_ax = fig.add_subplot(2, 1, 2)
    # box = img_ax.get_position()
    # img_ax.set_position([box.x0, box.y0 - box.height * 0.1, box.width,
    #                      box.height * 0.9])
    # Give each label its name (labels are from 0 to n_classes-1)
    lbr = SensitiveLabeler(label_name=[str(lbl) for lbl in range(n_classes)],
                           paragraphs=paragraphs)

    for i in range(quota):
        # ask_id = qs.make_query()
        # print("asking sample from Uncertainty Sampling")
        # # reshape the image to its width and height
        # lb = lbr.label(trn_ds.data[ask_id][0].reshape(8, 8))
        # trn_ds.update(ask_id, lb)
        # model.train(trn_ds)
        # E_out1 = np.append(E_out1, 1 - model.score(tst_ds))

        ask_id = qs2.make_query()
        print("asking sample from Random Sample")
        lb = lbr.label(trn_ds2.data[ask_id], ask_id)
        trn_ds2.update(ask_id, lb)
        model.train(trn_ds2)
            # E_out2 = np.append(E_out2, 1 - model.score(tst_ds))
            #
            # ax.set_xlim((0, i + 1))
            # ax.set_ylim((0, max(max(E_out1), max(E_out2)) + 0.2))
            # query_num = np.arange(0, i + 2)
            # p1.set_xdata(query_num)
            # p1.set_ydata(E_out1)
            # p2.set_xdata(query_num)
            # p2.set_ydata(E_out2)
            #
            # plt.draw()

    raw_input("Press any key to continue...")

if __name__ == '__main__':
    main()
