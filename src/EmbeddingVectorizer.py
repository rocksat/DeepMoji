from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class CountVectorizer(object):
    def __init__(self, use_tfidf=True):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()
        self.word_dictionary = {}
        self.use_tfidf = use_tfidf

    def load(self, word_dictionary):
        self.word_dictionary = word_dictionary

    def get_words(self, message):
        """Get the normalized list of words from a message string.

        This function should split a message into words, normalize them, and return
        the resulting list. For splitting, you should split on spaces. For
        normalization, you should convert everything to lowercase.

        Args:
            message: A string containing an SMS message

        Returns:
           The list of normalized words from the message.
        """
        words = message.strip().split()
        norm_words = [word.lower() for word in words]

        # apply stop words
        nonstop_words = [
            word for word in norm_words if not word in self.stop_words
        ]

        # apply stemming
        stem_words = [self.ps.stem(word) for word in nonstop_words]

        return stem_words

    def fit(self, messages, y=None):
        """Create a dictionary mapping words to integer indices.

        This function should create a dictionary of word to indices using the
        provided training messages. Use get_words to process each message.

        Rare words are often not useful for modeling. Please only add words to the
        dictionary if they occur in at least five messages.

        Args:
            messages: A list of strings containing SMS messages

        Returns:
            A python dict mapping words to integers.
        """
        # count word frequency
        word_frequency = {}
        for message in messages:
            words = self.get_words(message)
            for word in words:
                if word not in word_frequency:
                    word_frequency[word] = 1
                else:
                    word_frequency[word] += 1

        # build word dictionary
        word_list = []
        for index, (word, value) in enumerate(word_frequency.items()):
            if value >= 5:
                word_list.append(word)

        word_list.sort()
        self.word_dictionary = dict(zip(word_list, range(len(word_list))))

    @staticmethod
    def transform_tfidf(text_matrix):
        """Transform a count matrix to a normalized tf or tf-idf representation"""
        n, d = text_matrix.shape
        text_tfidf_matrix = np.zeros((n, d))

        # compute tf
        text_tf_matrix = text_matrix / (
            np.sum(text_matrix, axis=1, keepdims=True) + np.finfo('d').eps)

        # compute idf
        idf = n / (np.sum((text_matrix > 0.5) * 1, axis=0, keepdims=True) +
                   np.finfo('d').eps)
        text_tfidf_matrix = text_tf_matrix * np.log(idf)

        return text_tfidf_matrix

    def transform(self, messages):
        """Transform a list of text messages into a numpy array for further
        processing.

        This function should create a numpy array that contains the number of times
        each word of the vocabulary appears in each message.
        Each row in the resulting array should correspond to each message
        and each column should correspond to a word of the vocabulary.

        Use the provided word dictionary to map words to column indices. Ignore
        words that are not present in the dictionary. Use get_words to get the
        words for a message.

        Args:
            messages: A list of strings where each string is an SMS message.
            word_dictionary: A python dict mapping words to integers.

        Returns:
            A numpy array marking the words present in each message.
            Where the component (i,j) is the number of occurrences of the
            j-th vocabulary word in the i-th message.
        """
        n = len(messages)
        d = len(self.word_dictionary)
        text_matrix = np.zeros((n, d))

        # build text matrix
        for index, message in enumerate(messages):
            words = self.get_words(message)
            for word in words:
                if word in self.word_dictionary:
                    text_matrix[index, self.word_dictionary[word]] += 1

        if self.use_tfidf:
            text_matrix = self.transform_tfidf(text_matrix)

        return text_matrix

    def fit_transform(self, messages):
        self.fit(messages)
        return self.transform(messages)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)],
                    axis=0) for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf,
                                       [(w, tfidf.idf_[i])
                                        for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([
                self.word2vec[w] * self.word2weight[w]
                for w in words if w in self.word2vec
            ] or [np.zeros(self.dim)],
                    axis=0) for words in X
        ])
