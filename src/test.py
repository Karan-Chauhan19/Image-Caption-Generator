"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from config.config import Config
from nn_arch.attention import Attention
from nltk.translate.bleu_score import corpus_bleu

BASE_DIR = Config().TRAIN_CAPTIONS_PATH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, max_length, embedding_dim=512, hidden_dim=512):
        super(CaptionModel, self).__init__()

        # Encoder
        self.encoder_fc = nn.Linear(4096, hidden_dim)
        self.encoder_dropout = nn.Dropout(0.5)
        self.encoder_relu = nn.ReLU()

        # Decoder with Attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(hidden_dim + embedding_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_features, captions):
        # img_features: (batch_size, 4096)
        # captions: (batch_size, max_length)

        # Encode image features
        encoder_outputs = self.encoder_relu(self.encoder_fc(self.encoder_dropout(img_features)))
        encoder_outputs = encoder_outputs.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Embed captions
        embedded = self.embedding(captions)  # (batch_size, max_length, embedding_dim)

        # Initialize LSTM state
        batch_size = img_features.size(0)
        hidden = torch.zeros(1, batch_size, 512).to(device)
        cell = torch.zeros(1, batch_size, 512).to(device)

        outputs = []
        for t in range(captions.size(1)):
            # Apply attention
            context, _ = self.attention(hidden.squeeze(0), encoder_outputs)  # context: (batch_size, 1, hidden_dim)
            lstm_input = torch.cat((context, embedded[:, t, :].unsqueeze(1)), dim=2)  # (batch_size, 1, hidden_dim + embedding_dim)

            # LSTM step
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            output = self.dropout(output)
            output = self.fc(output.squeeze(1))
            outputs.append(output)

        return torch.stack(outputs, dim=1)

# Tokenizer Definition
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
        self.word2idx['<PAD>'] = 0
        self.idx2word[0] = '<PAD>'

    def texts_to_sequences(self, texts):
        return [[self.word2idx[word] for word in text.split() if word in self.word2idx] for text in texts]

    def get_vocab_size(self):
        return len(self.word2idx) + 1

    def index_to_word(self, index):
        return self.idx2word.get(index)

# Caption Processor Definition
class CaptionProcessor:
    def __init__(self, caption_file):
        self.caption_file = caption_file
        self.mapping = {}

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
            if image_id not in self.mapping:
                self.mapping[image_id] = []
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

# Main Inference Class
class ImageCaptionInference:
    def __init__(self, model_path, features_path, device):
        self.device = device

        # Load features
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)

        # Load captions and tokenizer
        processor = CaptionProcessor(os.path.join(BASE_DIR, 'captions.txt'))
        self.mapping = processor.load_captions()
        processor.clean_captions()
        all_captions = processor.get_all_captions()

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(all_captions)
        self.vocab_size = 8768
        self.max_length = max(len(caption.split()) for caption in all_captions)

        # Reverse word index for fast lookup
        self.index_word = {index: word for word, index in self.tokenizer.word2idx.items()}

        # Load model
        self.model = CaptionModel(self.vocab_size, self.max_length).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Test split
        image_ids = list(self.mapping.keys())
        split = int(len(image_ids) * 0.05)
        self.test = image_ids[split:]

    def predict_caption(self, image_features):
        in_text = 'startseq'
        image_features = torch.FloatTensor(image_features).to(self.device)

        # 🔹 Ensure image_features has correct batch dimension
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        with torch.no_grad():
            for _ in range(self.max_length):
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                sequence = torch.LongTensor(sequence).unsqueeze(0).to(self.device)  # Ensure batch dim

                # 🔹 Fix padding to match expected input shape
                sequence = torch.nn.functional.pad(sequence,
                                                   (0, self.max_length - sequence.shape[1]),
                                                   value=0)

                output = self.model(image_features, sequence)
                yhat = torch.argmax(output[:, -1, :], dim=1)
                word = self.index_word.get(yhat.item())

                if word is None or word == 'endseq':
                    break
                in_text += ' ' + word
        return in_text

    def evaluate(self, limit=100):
        actual, predicted = [], []
        for i, key in enumerate(tqdm(self.test)):
            if i >= limit:
                break
            captions = self.mapping[key]
            y_pred = self.predict_caption(self.features[key]).split()
            actual_captions = [caption.split() for caption in captions]
            actual.append(actual_captions)
            predicted.append(y_pred)

        bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-2: {bleu2:.4f}")

    def generate_caption(self, image_name):
        image_id = image_name.split('.')[0]
        img_path = os.path.join(BASE_DIR, "Images", image_name)
        image = Image.open(img_path)

        print('---------------------Actual---------------------')
        for caption in self.mapping[image_id]:
            print(caption)

        y_pred = self.predict_caption(self.features[image_id])
        print('--------------------Predicted--------------------')
        print(y_pred)

        plt.imshow(image)
        plt.axis('off')
        plt.show()

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inferencer = ImageCaptionInference(
        model_path=os.path.join(Config().PATH_TO_SAVE_TRAINED_MODEL, 'best_model.pt'),
        features_path=os.path.join(Config().PATH_TO_SAVE_TRAINED_MODEL, 'features.pkl'),
        device=device
    )

    # Evaluate BLEU scores
    inferencer.evaluate(limit=100)

    # Generate and visualize caption
    inferencer.generate_caption("1000268201_693b08cb0e.jpg")

if __name__ == "__main__":
    main()
