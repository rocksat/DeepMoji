# CS 229 Final Project: Prediction Emoji from Sentence
# Author:
#
#   Chen Huang : chuang4@stanford.edu
#   Boyu (Bill) Zhang: bzhang99@stanford.edu
#   Xueying (Shirley) Xie: xueyingx@stanford.edu
#
# predict emoji given sentence

# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
import numpy as np
import pickle
import argparse
import util
import EmbeddingVectorizer as ev


def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepMoji Sentence Classification')
    parser.add_argument('--sentence',
                        '-s',
                        type=str,
                        required=True,
                        help='sentence to analysis')
    parser.add_argument('--model_file',
                        '-m',
                        type=str,
                        default='../models/nb.pkl',
                        help='path to save trained model')
    parser.add_argument('--dictionary_file',
                        '-d',
                        type=str,
                        default='../models/word_dictionary.json',
                        help='path to save word dictionary')
    parser.add_argument('--word_embedding',
                        '-w',
                        type=str,
                        default='bow',
                        choices=['bow', 'glove', 'word2vec'],
                        help='word embedding [bow, glove, word2vec]')
    parser.add_argument('--emoji_map',
                        '-e',
                        type=str,
                        default='../data/emoji_map_1791.csv',
                        help='path to emoji mapping file')

    args = parser.parse_args()
    return args


def main(args):
    # step 1: load model
    loaded_model = pickle.load(open(args.model_file, 'rb'))

    # step 2: word embedding
    if args.word_embedding == 'bow':
        word_dictionary = util.read_json(args.dictionary_file)
        vectorizer = ev.CountVectorizer(use_tfidf=True)
        vectorizer.load(word_dictionary)
        word_vec = vectorizer.transform([args.sentence])
    elif args.word_embedding == 'glove':
        pass
    else:
        NotImplementedError("word2vec is not implemented yet")

    # step 3: predict
    predict = loaded_model.predict(word_vec)

    # display result
    print("The emoji label is %d" % predict[0])


if __name__ == '__main__':
    args = parse_args()
    main(args)
