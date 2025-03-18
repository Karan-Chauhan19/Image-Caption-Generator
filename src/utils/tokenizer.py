"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def fit_on_texts(self, texts):
        words = set()
        for text in texts:
            words.update(text.split())
        self.word2idx = {word: idx + 1 for idx, word in enumerate(sorted(words))}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def texts_to_sequences(self, texts):
        return [[self.word2idx[word] for word in text.split() if word in self.word2idx] for text in texts]

    def get_vocab_size(self):
        return len(self.word2idx) + 1