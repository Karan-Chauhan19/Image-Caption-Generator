"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

from tqdm import tqdm
from config.config import Config
from collections import defaultdict
from sentence_transformers import SentenceTransformer

class EmbeddingHandler:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.sentence_transformer = SentenceTransformer(model_name)
        self.embedding_dim = 384

    def process_captions(self, captions_file):
        with open(captions_file, 'r') as f:
            next(f)
            captions_doc = f.read()

        mapping = defaultdict(list)
        for line in tqdm(captions_doc.split('\n')):
            tokens = line.split(',')
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0], ' '.join(tokens[1:])
            image_id = image_id.split('.')[0]
            mapping[image_id].append(caption)

        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i].lower()
                caption = ''.join(c for c in caption if c.isalpha() or c.isspace())
                caption = ' '.join(word for word in caption.split() if len(word) > 1)
                captions[i] = f'startseq {caption} endseq'

        return mapping

    def generate_embeddings(self, mapping):
        caption_embeddings = {}
        for img_id, captions in mapping.items():
            embeddings = self.sentence_transformer.encode(captions, convert_to_tensor=True)
            caption_embeddings[img_id] = embeddings.cpu()
        return caption_embeddings

    def get_max_length(self, mapping):
        all_captions = [caption for captions in mapping.values() for caption in captions]
        return max(len(caption.split()) for caption in all_captions)
    
    def get_all_caption(self, mapping):
        all_captions = [caption for captions in mapping.values() for caption in captions]
        return all_captions