'''
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader

class CaptionDataset(Dataset):
    def __init__(self, image_ids, mapping, features, caption_embeddings, max_length):
        self.image_ids = image_ids
        self.mapping = mapping
        self.features = features
        self.caption_embeddings = caption_embeddings
        self.max_length = max_length

    def __len__(self):
        return len(self.image_ids) * 5  # Assuming 5 captions per image

    def __getitem__(self, idx):
        img_idx = idx // 5
        cap_idx = idx % 5
        image_id = self.image_ids[img_idx]

        X1 = torch.FloatTensor(self.features[image_id][0])  # Image features
        caption_embedding = self.caption_embeddings[image_id][cap_idx]

        return X1, caption_embedding
