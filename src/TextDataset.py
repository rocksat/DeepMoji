# Pytorch Style Data Loader for text sentences and emoji

import numpy as np
from torch.utils.data.dataset import Dataset


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
            messages.append(message)
            labels.append(label)

    return messages, np.array(labels)


def get_words(message):
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
    return norm_words


def create_dictionary(messages):
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
        words = get_words(message)
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
    return dict(zip(word_list, range(len(word_list))))


def transform_text(messages, word_dictionary):
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
    d = len(word_dictionary)
    text_matrix = np.zeros((n, d))

    # build text matrix
    for index, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                text_matrix[index, word_dictionary[word]] += 1

    return text_matrix


class TextDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (string): Path to the text file with sentences and emoji labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.messages, self.labels = load_text_dataset(txt_file)
        self.word_dictionary = create_dictionary(self.messages)
        self.text_matrix = transform_text(self.messages, self.word_dictionary)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = self.text_matrix[idx, :]
        label = self.labels[idx]
        return message, label
