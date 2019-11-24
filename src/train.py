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

# customized library
from util import write_json
from TextDataset import TextDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepMoji Sentence Classification')

    parser.add_argument('--dataset',
                        '-d',
                        type=str,
                        default='../data/dataset.txt',
                        help='dataset path')
    parser.add_argument('--num_train',
                        '-n',
                        type=int,
                        default=20000,
                        help='number of samples for training')
    parser.add_argument('--classifier',
                        '-c',
                        type=str,
                        default='nb',
                        choices=['nb', 'svm', 'dnn'],
                        help='classifier type [nb, svm, dnn]')
    parser.add_argument('--save_path',
                        '-s',
                        type=str,
                        default='../models',
                        help='path to save trained model and word dictionary')
    return parser.parse_args()


def main(args):
    # step 1: load dataset
    dataset = TextDataset(args.dataset)
    dictionary_file = os.path.join(args.save_path, 'word_dictionary.json')
    write_json(dictionary_file, dataset.word_dictionary)

    # step 2: train / test split
    messages_train, messages_test = np.split(dataset.text_matrix,
                                             [args.num_train],
                                             axis=0)
    labels_train, labels_test = np.split(dataset.labels, [args.num_train],
                                         axis=0)

    # step 2: load classifier
    if args.classifier == 'nb':
        from sklearn.naive_bayes import MultinomialNB
        model_file = os.path.join(args.save_path, 'nb.pkl')
        clf = MultinomialNB()
    elif args.classifier == 'svm':
        from sklearn.svm import SVC
        model_file = os.path.join(args.save_path, 'svm.pkl')
        clf = SVC(gamma='auto', max_iter=10000)
    else:
        NotImplementedError("DNN classifier is not implemented yet")

    # step 3: train classifier
    clf.fit(messages_train, labels_train)
    pickle.dump(clf, open(model_file, 'wb'))

    # step 4: evaluate classifier
    accuracy = clf.score(messages_test, labels_test)
    print('%s classifier accuracy is %.3f%%' %
          (args.classifier, accuracy * 100))


if __name__ == '__main__':
    args = parse_args()
    main(args)