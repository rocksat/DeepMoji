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
import os
import pandas as pd
import seaborn as sn
import sklearn.model_selection as model_selection
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# customized library
from util import load_glove_model
from TextDataset import TextDataset

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
                        default='lstm',
                        choices=['cnn', 'lstm', 'gru'],
                        help='classifier type [cnn, lstm, gru]')
    parser.add_argument('--word_embedding',
                        '-w',
                        type=str,
                        default='glove-50',
                        choices=['glove-50', 'glove-300', 'bert'],
                        help='word embedding [glove-50, glove-300, bert]')
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
    if args.word_embedding == 'glove-50':
        glove_6_b_50_d_path = os.path.join(args.out_path,
                                           'pretrain_models/glove.6B.50d.txt')
        word_embedding = load_glove_model(glove_6_b_50_d_path)
        embedding_dim = 50
    elif args.word_embedding == 'glove-300':
        glove_6_b_300_d_path = os.path.join(
            args.out_path, 'pretrain_models/glove.6B.300d.txt')
        word_embedding = load_glove_model(glove_6_b_300_d_path)
        embedding_dim = 300
    else:
        NotImplementedError('not implemented yet')

    # step 4: train classifier
    if args.classifier == 'cnn':
        from CNNClassifier import CNNClassifier
        clf = CNNClassifier(messages=dataset.messages,
                            labels=dataset.labels,
                            word_embedding=word_embedding,
                            embedding_dim=embedding_dim)
    elif args.classifier == 'lstm':
        from LSTMClassifier import LSTMClassifier
        clf = LSTMClassifier(messages=dataset.messages,
                             labels=dataset.labels,
                             word_embedding=word_embedding,
                             embedding_dim=embedding_dim,
                             lstm_output_size=100,
                             use_attention_layer=False)
    else:
        from GRUClassifier import GRUClassifier
        clf = GRUClassifier(messages=dataset.messages,
                            labels=dataset.labels,
                            word_embedding=word_embedding,
                            embedding_dim=embedding_dim,
                            gru_output_size=100,
                            use_attention_layer=False)

    X_train, y_train = clf.data_preproccess(messages_train, labels_train)
    clf.fit(X_train, y_train, epochs=10)
    model_file = os.path.join(
        args.out_path, 'models/{}_{}.pkl'.format(args.word_embedding,
                                                 args.classifier))
    clf.save(model_file)

    # step 5: evaluate on test set
    X_test, y_test = clf.data_preproccess(messages_test, labels_test)
    scores = clf.score(X_test, y_test)
    print("%s classifier %s on %s is : %.3f%%" %
          (args.classifier, clf.model.metrics_names[1], args.word_embedding,
           scores[1] * 100))

    # step 6: draw confusion matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    df_cm = pd.DataFrame(cm,
                         index=[i for i in clf.le.classes_],
                         columns=[i for i in clf.le.classes_])
    plt.figure(figsize=(10, 7))
    plt.title('{}_{} confusion matrix'.format(args.word_embedding,
                                              args.classifier))
    sn.heatmap(df_cm, annot=False, cmap=plt.get_cmap('Blues'))
    figure_file = os.path.join(
        args.out_path, 'figure/{} {}.png'.format(args.word_embedding,
                                                 args.classifier))
    plt.savefig(figure_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
