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


def load_glove_model(filename):
    """load pretrained glove models"""
    embeddings_index = {}
    with open(filename, "rb") as lines:
        for line in lines:
            word, coefs = line.split(maxsplit=1)
            word = word.decode("utf-8")
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    return embeddings_index
