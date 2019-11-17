# CS 229 Final Project: Prediction Emoji from Sentence
# Author:
#
#   Chen Huang : chuang4@stanford.edu
#   Boyu (Bill) Zhang: bzhang99@stanford.edu
#   Xueying (Shirley) Xie: xueyingx@stanford.edu
#

# build-in library
import sys
import os
import argparse
from torch.utils.data import DataLoader

# customized library
from TextDataset import TextDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepMoji Sentence Classification')

    parser.add_argument('--dataset',
                        '-d',
                        type=str,
                        default='../data/dataset.txt',
                        help='dataset path')
    parser.add_argument('--classifier',
                        '-c',
                        type=str,
                        default='nb',
                        choices=['nb', 'svm', 'dnn'],
                        help='classifier type [nb, svm, dnn]')
    return parser.parse_args()


def main(args):
    # step 1: load dataset
    dataset = TextDataset(args.dataset)

    # step 2: load classifier
    if args.classifier == 'nb':
        from sklearn.naive_bayes import GaussianNB as Classifier
    elif args.classifier == 'svm':
        from sklearn import svm as Classifier
    else:
        NotImplementedError("DNN classifier is not implemented yet")

    # step 3: train classifier

    # step 3: evaluate classifier
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
