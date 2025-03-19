"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

from tqdm import tqdm
from config.config import Config
from collections import defaultdict

class CaptionProcessor:
    def __init__(self, caption_file=Config().TRAIN_CAPTIONS_PATH):
        self.caption_file = caption_file
        self.mapping = defaultdict(list)

    def load_captions(self):
        with open(self.caption_file, 'r') as f:
            next(f)
            captions_doc = f.read()
        
        for line in tqdm(captions_doc.split('\n')):
            tokens = line.split(',')
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0], ' '.join(tokens[1:])
            image_id = image_id.split('.')[0]
            self.mapping[image_id].append(caption)
        return self.mapping

    def clean_captions(self):
        for key, captions in self.mapping.items():
            for i in range(len(captions)):
                caption = captions[i].lower()
                caption = ''.join(c for c in caption if c.isalpha() or c.isspace())
                caption = ' '.join(word for word in caption.split() if len(word) > 1)
                captions[i] = f'startseq {caption} endseq'

    def get_all_captions(self):
        return [caption for captions in self.mapping.values() for caption in captions]