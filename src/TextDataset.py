# Pytorch Style Data Loader for text sentences and emoji

import numpy as np
from torch.utils.data.dataset import Dataset
import util


class TextDataset(Dataset):
    def __init__(self,
                 txt_file,
                 occurrence,
                 top_k_stop_emoji,
                 max_messages=None):
        """
        Args:
            txt_file (string): Path to the text file with sentences and emoji labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.messages, self.labels = util.load_text_dataset(txt_file)
        self.filter(occurrence=occurrence, top_k_stop_emoji=top_k_stop_emoji)
        self.length = len(self.messages)

        if max_messages and max_messages > self.length:
            shuffle_index = np.arange(self.length)
            np.random.shuffle(shuffle_index)
            self.messages = [
                self.messages[i] for i in shuffle_index[:max_messages]
            ]
            self.labels = self.labels[shuffle_index[:max_messages]]

    def filter(self, occurrence, top_k_stop_emoji):
        sample_count = len(self.messages)
        labels_count = np.bincount(self.labels)

        if top_k_stop_emoji:
            # remove top 5 stop emoji from label_count
            for i in range(top_k_stop_emoji):
                stop_emoji = np.argmax(labels_count)
                labels_count[stop_emoji] = -1

        select_masks = labels_count[self.labels] > occurrence
        self.messages = [
            self.messages[i] for i in range(sample_count) if select_masks[i]
        ]
        self.labels = self.labels[select_masks]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        message = self.messages[idx, :]
        label = self.labels[idx]
        return message, label
