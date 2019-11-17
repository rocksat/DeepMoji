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
import pandas as pd
import emoji


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
                        required=True,
                        default='../models/nb.pkl',
                        help='path to save trained model')
    parser.add_argument('--dictionary_file',
                        '-d',
                        type=str,
                        required=True,
                        default='../models/word_dictionary.txt',
                        help='path to save word dictionary')
    parser.add_argument('--emoji_map',
                        '-e',
                        type=str,
                        required=True,
                        default='../data/emoji_map_1791.csv',
                        help='path to emoji mapping file')

    args = parser.parse_args()
    return args


def main(args):
    # step 1: load model
    loaded_model = pickle.load(open(args.model_file, 'rb'))

    # step 2: load word dictionary
    word_dictionary = util.read_json(args.dictionary_file)

    # step 3: pass sentence into matrix
    word_vec = util.transform_text([args.sentence], word_dictionary)

    # step 4: predict
    predict = loaded_model.predict(word_vec)

    # display results
    emoji = pd.read_csv(args.emoji_map, names=['ucode'], encoding='utf-8')
    print(emoji[predict[0] - 1])


if __name__ == '__main__':
    args = parse_args()
    main(args)
