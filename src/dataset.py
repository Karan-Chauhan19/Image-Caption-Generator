'''
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader

class ImageCaptionDataset(Dataset):
    def __init__(self, image_ids, mapping, features, tokenizer, max_length):
        self.image_ids = image_ids
        self.mapping = mapping
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_ids) * 5

    def __getitem__(self, idx):
        img_idx = idx // 5
        cap_idx = idx % 5
        image_id = self.image_ids[img_idx]
        caption = self.mapping[image_id][cap_idx]
        seq = self.tokenizer.texts_to_sequences([caption])[0]

        X1 = torch.FloatTensor(self.features[image_id][0])
        X2 = torch.LongTensor(seq[:-1])
        y = torch.LongTensor(seq[1:])

        X2 = torch.nn.functional.pad(X2, (0, self.max_length - len(X2)), value=0)
        y = torch.nn.functional.pad(y, (0, self.max_length - len(y)), value=0)

        return X1, X2, y