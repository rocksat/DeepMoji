# Pytorch Style Data Loader for text sentences and emoji

import numpy as np
from torch.utils.data.dataset import Dataset
import util


class TextDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (string): Path to the text file with sentences and emoji labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.messages, self.labels = util.load_text_dataset(txt_file)
        self.word_dictionary = util.create_dictionary(self.messages)
        self.text_matrix = util.transform_text(self.messages,
                                               self.word_dictionary)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = self.text_matrix[idx, :]
        label = self.labels[idx]
        return message, label
