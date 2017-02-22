#!/usr/bin/env python3
# This is an example run of LDA on State of the Union addresses given by past
#   US presidents. It sort of follows the scikit-learn example given at:
#   http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html

from os import listdir
from os.path import isfile, join, sep

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# These are just some useful constants
n_samples = 2000
n_features = 1000
n_topics = 100
n_top_words = 50


def print_top_words(model, feature_names, n_top_words):
    """Print some meaningful output from LDA"""
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
#might be this "topic", print to pickle file.
    print()


def get_paragraphs(filenames):
    """Get the paragraphs from a list of files"""
    # I create a list for the file contents to go in
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


# First we load the data
# I make a list of all filenames in the 'sotu' directory using a list
#   comprehension. I can explain more in person, or you can look it up. Also,
#   'sep' as used below is the system-specific path separator, normally / or \
filenames = ['test2'+sep+f for f in listdir('test2')
             if isfile(join('test2', f))]
paragraphs = get_paragraphs(filenames)

# The data is now loaded into the program, so we perform setup for LDA
# The CountVectorizer turns the documents from a sequence of words to a vector
#   of numbers.
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english')
# We use our CountVectorizer to transform the paragraphs into vectors.
tf = tf_vectorizer.fit_transform(paragraphs)

# We then create our LDA model
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
# We train LDA on our data
lda.fit(tf)

# We then print out the top 20 words of each topic
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
