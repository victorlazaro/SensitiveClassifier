#!/usr/bin/env python3
"""
This script compares random sampling versus uncertainty sampling, so that we
can see how much faster uncertainty sampling improves than random sampling.
"""
import copy
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.models import LogisticRegression, SVM
from libact.query_strategies import UncertaintySampling, RandomSampling


def preprocessing(paragraphs):
    """Performs birth year preprocessing"""
    import re
    processed = []
    for paragraph in paragraphs:
        processed.append(re.sub(r'(?<![0-9])((([0-2][0-9][0-9][0-9])|([0-9][0-9][0-9])|([0-9][0-9])|([0-9])))(?![0-9])', "possibleYear", paragraph))
    return processed


def get_paragraphs():
    """Get the paragraphs from a list of files"""
    docs_non = []
    docs_sens = []
    non_sensitive_file = "test-non"
    sensitive_file = "test-sensitive"

    if isfile(non_sensitive_file):
        with open(non_sensitive_file, 'r') as doc:
            paragraphs = [paragraph for paragraph in doc.read().split('\n')]
            docs_non.extend(paragraphs)

    if isfile(sensitive_file):
        with open(sensitive_file, 'r') as doc:
            paragraphs = [paragraph for paragraph in doc.read().split('\n')]
            docs_sens.extend(paragraphs)

    docs_non = preprocessing(docs_non)
    docs_sens = preprocessing(docs_sens)

    return docs_non, docs_sens

def get_sets(n_topics, max_iters, learning_offset):
    """Splits the data into training and test sets"""
    # Some hyperparameters
    n_features = 1000

    docs_non, docs_sen = get_paragraphs()
    mixed = []
    mixed.extend(docs_non)
    mixed.extend(docs_sen)

    # Turn the documents into word counts
    tf_vect = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                        stop_words='english')
    tf = tf_vect.fit_transform(mixed)
    # Turn word counts into topics
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=max_iters,
                                    learning_method='online',
                                    learning_offset=learning_offset)
    lda.fit(tf)
    # Get the data into train and test sets
    X = lda.transform(tf)
    y = [0] * len(docs_non)
    y.extend([1] * len(docs_sen))

    return X, y

def split_train_test(num_labeled, X, y):
    # """Splits the data into training and test sets"""
    # # Some hyperparameters
    # n_topics = 100
    # n_features = 1000
    #
    # docs_non, docs_sen = get_paragraphs()
    # mixed = []
    # mixed.extend(docs_non)
    # mixed.extend(docs_sen)
    #
    # # Turn the documents into word counts
    # tf_vect = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
    #                                     stop_words='english')
    # tf = tf_vect.fit_transform(mixed)
    # # Turn word counts into topics
    # lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
    #                                 learning_method='online',
    #                                 learning_offset=50.,
    #                                 random_state=0)
    # lda.fit(tf)
    # # Get the data into train and test sets
    # X = lda.transform(tf)
    # y = [0] * len(docs_non)
    # y.extend([1] * len(docs_sen))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


    y_train_unlabeled = []

    y_train_unlabeled.extend(y_train[:num_labeled])

    while(0 not in y_train_unlabeled or 1 not in y_train_unlabeled):
        y_train_unlabeled.append(y_train[len(y_train_unlabeled)])

    y_train_unlabeled.extend([None] * len(y_train))
    train_ds = Dataset(X_train, y_train_unlabeled)
    test_ds = Dataset(X_test, y_test)
    labeled_train_ds = Dataset(X_train, y_train)

    return train_ds, test_ds, y_train, labeled_train_ds


def _run(train_ds, test_ds, labeler, model, query_strat, quota):
    """Runs the active learning process"""
    train_error, test_error = [], []

    for _ in range(quota):
        ask_id = query_strat.make_query()
        X, _ = zip(*train_ds.data)
        label = labeler.label(X[ask_id])
        train_ds.update(ask_id, label)

        model.train(train_ds)
        train_error = np.append(train_error, 1 - model.score(train_ds))
        test_error = np.append(test_error, 1 - model.score(test_ds))

    return train_error, test_error

def main(X, y):
    """Runs the main program"""
    # Start out with 10 instances labeled, so uncertainty sampling will work
    num_labeled = 10

    train_ds, test_ds, y_train, labeled_train_ds = split_train_test(num_labeled, X, y)



    train_ds2 = copy.deepcopy(train_ds)
    labeler = IdealLabeler(labeled_train_ds)

    quota = len(y_train) - num_labeled

    uncertain_strat = UncertaintySampling(train_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    train_error_uncertainty, test_error_uncertainty = _run(train_ds,
                                                           test_ds,
                                                           labeler,
                                                           model,
                                                           uncertain_strat,
                                                           quota)

    random_strat = RandomSampling(train_ds2)
    model = LogisticRegression()
    train_error_random, test_error_random = _run(train_ds2,
                                                 test_ds,
                                                 labeler,
                                                 model,
                                                 random_strat,
                                                 quota)

    # query_num = np.arange(1, quota+1)
    # plt.plot(query_num, train_error_uncertainty, 'b', label='Uncertainty Train')
    # plt.plot(query_num, test_error_uncertainty, 'r', label='Uncertainty Test')
    # plt.plot(query_num, train_error_random, 'g', label='Random Train')
    # plt.plot(query_num, test_error_random, 'k', label='Random Test')
    # plt.xlabel('Number of Queries')
    # plt.ylabel('Error')
    # plt.title('Uncertainty Sampling vs Random Sampling')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=5)
    # plt.show()
    return test_error_uncertainty, test_error_random, quota


def simple(X, y):
    num_experiments = 100
    print('Running the experiment 100 times and averaging the results...')
    print('Please wait...')
    best_accuracy_random = 0
    best_accuracy_uncertainty = 0
    uncertain_test_errors, random_test_errors, quota = main(X, y)
    for i in range(num_experiments - 1):
        if (i + 1) % 10 == 0:
            print('finished', str(i + 1), 'experiments...')
        test_error_uncertainty, test_error_random, _ = main(X, y)
        # This should add the elements of the arrays together element-wise
        uncertain_test_errors += test_error_uncertainty
        random_test_errors += test_error_random

    # This should divide the elements of the array element-wise
    uncertain_test_errors /= 100
    random_test_errors /= 100
    best_accuracy_random = 1 - min(random_test_errors)
    best_accuracy_uncertainty = 1 - min(uncertain_test_errors)

    return best_accuracy_uncertainty, best_accuracy_random

    # query_num = np.arange(1, quota + 1)
    # plt.plot(query_num, uncertain_test_errors, 'r', label='Uncertainty Test')
    # plt.plot(query_num, random_test_errors, 'k', label='Random Test')
    # plt.xlabel('Number of Queries')
    # plt.xticks(range(0, 100, 10), range(10,100,10))
    # plt.ylabel('Error')
    # plt.title('Uncertainty Sampling vs Random Sampling')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=5)
    # plt.show()

if __name__ == '__main__':

    min_topics = 20
    max_topics = 101
    topics_step_size = 20
    min_iters = 2
    max_iters = 30
    iters_step_size = 3

    min_offset = 5
    max_offset = 30
    offset_step_size = 5

    lda_hyperaparams = {'n_topics': [i for i in range(min_topics, max_topics, topics_step_size)],
                        'max_iters': [i for i in range(min_iters, max_iters, iters_step_size)],
                        'learning_offset': [i for i in range(min_offset, max_offset, offset_step_size)]}

    x_axis = []
    random_accuracies = []
    uncertainty_accuracies = []


    for n_topics in lda_hyperaparams['n_topics']:
        for max_iter in lda_hyperaparams['max_iters']:
            for learning_offset in lda_hyperaparams['learning_offset']:
                print('Running with n_topics:', n_topics, 'max_iter:', max_iter, 'learning_offset:', learning_offset)
                X, y = get_sets(n_topics, max_iter, learning_offset)
                best_accuracy_uncertainty, best_accuracy_random = simple(X, y)
                random_accuracies.append(best_accuracy_random)
                uncertainty_accuracies.append(best_accuracy_uncertainty)
                x_axis.append(str(n_topics) + ' ' + str(max_iter) + ' ' + str(learning_offset))
    plt.plot(x_axis, uncertainty_accuracies, 'r', label='Uncertainty Tests')
    plt.plot(x_axis, random_accuracies, 'k', label='Random Tests')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Accuracy')
    plt.title('Hyperparameter analysis')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()

    # OPTIMAL NUM_TOPICS IS 60
    # for index, n_topics in enumerate(lda_hyperaparams['n_topics']):
    #     print('This is the', index + 1, 'parameter testing')
    #     X, y = get_sets(n_topics)
    #     best_accuracy_uncertainty, best_accuracy_random = simple(X, y)
    #     random_accuracies.append(best_accuracy_random)
    #     uncertainty_accuracies.append(best_accuracy_uncertainty)
    # plt.plot(lda_hyperaparams['n_topics'], uncertainty_accuracies, 'r', label='Uncertainty Tests')
    # plt.plot(lda_hyperaparams['n_topics'], random_accuracies, 'k', label='Random Tests')
    # plt.xlabel('Number of Topics')
    # plt.ylabel('Accuracy')
    # plt.title('Num Topics')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=5)
    # plt.show()


    # OPTIMAL MAX_ITERS IS 26
    # for index, iters in enumerate(lda_hyperaparams['max_iters']):
    #     print('This is the', index + 1, 'parameter testing')
    #     X, y = get_sets(60, iters)
    #     best_accuracy_uncertainty, best_accuracy_random = simple(X, y)
    #     random_accuracies.append(best_accuracy_random)
    #     uncertainty_accuracies.append(best_accuracy_uncertainty)
    # plt.plot(lda_hyperaparams['max_iters'], uncertainty_accuracies, 'r', label='Uncertainty Tests')
    # plt.plot(lda_hyperaparams['max_iters'], random_accuracies, 'k', label='Random Tests')
    # plt.xlabel('Max Iterations')
    # plt.ylabel('Accuracy')
    # plt.title('Max Iters')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=5)
    # plt.show()

    # OPTIMAL LEARNING_OFFSET IS 20
    # for index, learning_offset in enumerate(lda_hyperaparams['learning_offset']):
    #     print('This is the', index + 1, 'parameter testing')
    #     X, y = get_sets(60, 26, learning_offset)
    #     best_accuracy_uncertainty, best_accuracy_random = simple(X, y)
    #     random_accuracies.append(best_accuracy_random)
    #     uncertainty_accuracies.append(best_accuracy_uncertainty)
    # plt.plot(lda_hyperaparams['learning_offset'], uncertainty_accuracies, 'r', label='Uncertainty Tests')
    # plt.plot(lda_hyperaparams['learning_offset'], random_accuracies, 'k', label='Random Tests')
    # plt.xlabel('Learning Offset')
    # plt.ylabel('Accuracy')
    # plt.title('Learning Offset')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=5)
    # plt.show()


