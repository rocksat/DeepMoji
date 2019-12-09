import numpy as np
import json


def load_text_dataset(txt_file):
    """Load the deep emoji dataset from a text file

    Args:
         txt_file: Path to text file containing sentences and emoji labels.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The emoji labels (0 or 1) for each message.
    """
    messages = []
    labels = []

    with open(txt_file, 'r', newline='', encoding='utf8') as f:
        lines = f.readlines()

        for line in lines:
            message, label = line.split(':')
            messages.append(message.strip())
            labels.append(int(label.strip()))

    return messages, np.array(labels)


def transform_tfidf(text_matrix, use_idf=True):
    """Transform a count matrix to a normalized tf or tf-idf representation"""
    n, d = text_matrix.shape
    text_tfidf_matrix = np.zeros((n, d))

    # compute tf
    text_tf_matrix = text_matrix / (
        np.sum(text_matrix, axis=1, keepdims=True) + np.finfo('d').eps)

    # compute idf
    idf = n / (np.sum(
        (text_matrix > 0.5) * 1, axis=0, keepdims=True) + np.finfo('d').eps)
    text_tfidf_matrix = text_tf_matrix * np.log(idf)

    return text_tfidf_matrix if use_idf else text_tf_matrix


def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)


def read_json(filename):
    """Read the provided JSON file into dictionary"""
    with open(filename, 'r') as f:
        value = json.load(f)
    return value


def load_emoji(filename):
    """Read label to emoji label"""
    # Load headers
    with open(filename, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
