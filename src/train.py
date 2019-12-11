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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

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
    parser.add_argument('--out_path',
                        '-o',
                        type=str,
                        default='../artifact',
                        help='path to save trained model and word dictionary')
    return parser.parse_args()


def main(args):
    # step 1: load dataset
    dataset = TextDataset(args.dataset, occurrence=1000, top_k_stop_emoji=30)

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
        dictionary_file = os.path.join(args.out_path,
                                       'models/word_dictionary.json')
        write_json(dictionary_file, vectorizer.word_dictionary)
    elif args.word_embedding == 'glove':
        # load pre-trained glove model
        glove_6_b_50_d_path = os.path.join(args.out_path,
                                           'pretrain_models/glove.6B.50d.txt')
        glove_model = load_glove_model(glove_6_b_50_d_path)
        vectorizer = ev.MeanEmbeddingVectorizer(glove_model)
    # encoder labels
    le = LabelEncoder()
    le.fit(dataset.labels)

    X_train = vectorizer.transform(messages_train)
    X_test = vectorizer.transform(messages_test)
    y_train = le.transform(labels_train)
    y_test = le.transform(labels_test)

    # step 4: train classifier
    n_estimators = 5
    if args.classifier == 'nb':
        from sklearn.naive_bayes import MultinomialNB
        clf = BaggingClassifier(MultinomialNB(),
                                max_samples=1.0 / n_estimators,
                                n_estimators=n_estimators,
                                n_jobs=-1,
                                verbose=False)
    elif args.classifier == 'svm':
        from sklearn.svm import SVC
        clf = BaggingClassifier(SVC(gamma='auto'),
                                max_samples=1.0 / n_estimators,
                                n_estimators=n_estimators,
                                n_jobs=-1,
                                verbose=False)
    else:
        NotImplementedError('not implemented yet')

    clf.fit(X_train, y_train)
    model_file = os.path.join(
        args.out_path, 'models/{}_{}.pkl'.format(args.word_embedding,
                                                 args.classifier))
    pickle.dump(clf, open(model_file, 'wb'))

    # step 5: evaluate on test set
    eval_set = {'train_set': (X_train, y_train), 'test_set': (X_test, y_test)}
    for t, (X, y) in eval_set.items():
        accuracy = clf.score(X, y)
        print('%s classifier accuracy on %s is %.3f%%' %
              (args.classifier, t, accuracy * 100))

    # step 6: plot confusion matrix
    disp = plot_confusion_matrix(clf,
                                 X_test,
                                 y_test,
                                 display_labels=le.classes_,
                                 include_values=False,
                                 cmap=plt.get_cmap('Blues'),
                                 normalize='true')
    disp.ax_.set_title('{}_{} confusion matrix'.format(args.word_embedding,
                                                       args.classifier))
    figure_file = os.path.join(
        args.out_path, 'figure/{}_{}.png'.format(args.word_embedding,
                                                 args.classifier))
    plt.savefig(figure_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
