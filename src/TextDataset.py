# Pytorch Style Data Loader for text sentences and emoji

import numpy as np
from torch.utils.data.dataset import Dataset
import util


class TextDataset(Dataset):
    def __init__(self, txt_file, max_messages=50000):
        """
        Args:
            txt_file (string): Path to the text file with sentences and emoji labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.messages, self.labels = util.load_text_dataset(txt_file)
        self.filter()

        if max_messages:
            total_messages = len(self.messages)
            shuffle_index = np.arange(total_messages)
            np.random.shuffle(shuffle_index)
            self.messages = [self.messages[i] for i in shuffle_index[:max_messages]]
            self.labels = self.labels[shuffle_index[:max_messages]]

    def filter(self, occurrence=500):
        sample_count = len(self.messages)
        labels_count = np.bincount(self.labels)
        select_masks = labels_count[self.labels] > occurrence
        self.messages = [
            self.messages[i] for i in range(sample_count) if select_masks[i]
        ]
        self.labels = self.labels[select_masks]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = self.messages[idx, :]
        label = self.labels[idx]
        return message, label
