# CS 229 Final Project: Prediction Emoji from Sentence
# Author:
#
#   Chen Huang : chuang4@stanford.edu
#   Boyu (Bill) Zhang: bzhang99@stanford.edu
#   Xueying (Shirley) Xie: xueyingx@stanford.edu
#

# build-in library
import numpy as np
import argparse
import pickle
import os
import sklearn.model_selection as model_selection
from sklearn.ensemble import BaggingClassifier

# customized library
from util import write_json, load_glove_model
from TextDataset import TextDataset
import EmbeddingVectorizer as ev

random_state = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepMoji Sentence Classification')

    parser.add_argument('--dataset',
                        '-d',
                        type=str,
                        default='../data/dataset.txt',
                        help='dataset path')
    parser.add_argument('--test_size',
                        '-t',
                        type=float,
                        default=0.2,
                        help='the proportion of the dataset in the test')
    parser.add_argument('--classifier',
                        '-c',
                        type=str,
                        default='nb',
                        choices=['nb', 'svm', 'dnn'],
                        help='classifier type [nb, svm, dnn]')
    parser.add_argument('--word_embedding',
                        '-w',
                        type=str,
                        default='bow',
                        choices=['bow', 'glove', 'word2vec'],
                        help='word embedding [bow, glove, word2vec]')

    parser.add_argument('--save_path',
                        '-s',
                        type=str,
                        default='../models',
                        help='path to save trained model and word dictionary')
    return parser.parse_args()


def main(args):
    # step 1: load dataset
    dataset = TextDataset(args.dataset)

    # step 2: train / test split
    messages_train, messages_test, labels_train, labels_test = model_selection.train_test_split(
        dataset.messages,
        dataset.labels,
        test_size=args.test_size,
        random_state=random_state)

    # step 3: word embedding
    if args.word_embedding == 'bow':
        vectorizer = ev.CountVectorizer(use_tfidf=True)
        vectorizer.fit(dataset.messages)
        dictionary_file = os.path.join(args.save_path, 'word_dictionary.json')
        write_json(dictionary_file, vectorizer.word_dictionary)
    elif args.word_embedding == 'glove':
        # load pre-trained glove model
        glove_6_b_50_d_path = os.path.join(args.save_path, 'glove.6B.5d.txt')
        glove_model = load_glove_model(glove_6_b_50_d_path)
        vectorizer = ev.MeanEmbeddingVectorizer(glove_model)

    X_train = vectorizer.transform(messages_train)
    X_test = vectorizer.transform(messages_test)

    # step 4: train classifier
    n_estimators = 10
    if args.classifier == 'nb':
        from sklearn.naive_bayes import MultinomialNB
        clf = BaggingClassifier(MultinomialNB(),
                                max_samples=1.0 / n_estimators,
                                n_estimators=n_estimators,
                                n_jobs=-1,
                                verbose=True)
    elif args.classifier == 'svm':
        from sklearn.svm import SVC
        clf = BaggingClassifier(SVC(gamma='auto'),
                                max_samples=1.0 / n_estimators,
                                n_estimators=n_estimators,
                                n_jobs=-1,
                                verbose=True)
    else:
        NotImplementedError("DNN classifier is not implemented yet")

    clf.fit(X_train, labels_train)
    model_file = os.path.join(
        args.save_path, '{}_{}.pkl'.format(args.word_embedding,
                                           args.classifier))
    pickle.dump(clf, open(model_file, 'wb'))

    # step 5: evaluate on testset
    accuracy = clf.score(X_test, labels_test)
    print('%s classifier accuracy is %.3f%%' %
          (args.classifier, accuracy * 100))


if __name__ == '__main__':
    args = parse_args()
    main(args)
