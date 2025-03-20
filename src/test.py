"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

import os
import torch
import pickle
import numpy as np
from PIL import Image
import torch.nn as nn
from nn_arch.vgg16 import VGG16
from config.config import Config
from torchvision import transforms
from collections import defaultdict
from utils.embeddings import EmbeddingHandler
from sentence_transformers import SentenceTransformer

# Load the trained caption model (corrected version)
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, hidden, encoder_outputs):
        energy = self.tanh(self.attn(torch.cat((hidden, encoder_outputs.squeeze(1)), dim=1)))
        attention_weights = self.softmax(torch.matmul(energy, self.v))
        context = attention_weights.unsqueeze(1).unsqueeze(2) * encoder_outputs
        return context, attention_weights
    
class CaptionModel(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=512):
        super(CaptionModel, self).__init__()
        self.encoder_fc = nn.Linear(4096, hidden_dim)
        self.encoder_dropout = nn.Dropout(0.5)
        self.encoder_relu = nn.ReLU()
        self.embedding_fc = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, img_features, caption_embeddings=None):
        # img_features: (batch_size, 4096)
        # caption_embeddings: (batch_size, embedding_dim) or None for inference

        encoder_outputs = self.encoder_relu(self.encoder_fc(self.encoder_dropout(img_features)))
        encoder_outputs = encoder_outputs.unsqueeze(1)

        batch_size = img_features.size(0)
        hidden = torch.zeros(1, batch_size, 512).to(device)
        cell = torch.zeros(1, batch_size, 512).to(device)

        if caption_embeddings is None:
            # For inference, start with a zero embedding
            caption_embedding = torch.zeros(batch_size, 384).to(device)
        else:
            caption_embedding = caption_embeddings

        embedded = self.embedding_fc(caption_embedding).unsqueeze(1)
        context, _ = self.attention(hidden.squeeze(0), encoder_outputs)
        lstm_input = torch.cat((context, embedded), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # Fixed: replaced 'opgesloten' with 'cell'
        output = self.dropout(output)
        output = self.fc(output.squeeze(1))

        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(img_path, feature_extractor, all_captions, all_caption_embeddings):

    # Load pretrained model
    model = CaptionModel(embedding_dim=384).to(device)
    model.load_state_dict(torch.load(os.path.join(Config().PATH_TO_SAVE_TRAINED_MODEL, 'best_model.pt'), map_location=torch.device('cpu')))
    model.eval()

    # Load and preprocess image
    image = Image.open(img_path).convert('RGB')
    image = VGG16(device).preprocess(image).unsqueeze(0).to(device)

    # Extract image features
    with torch.no_grad():
        img_features = feature_extractor(image)

    # Generate caption embedding
    with torch.no_grad():
        predicted_embedding = model(img_features).cpu()

    # Find nearest caption
    similarities = torch.cosine_similarity(predicted_embedding, all_caption_embeddings)
    best_match_idx = similarities.argmax()
    generated_caption = all_captions[best_match_idx]

    return generated_caption

# Load Sentence Transformer
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# Load training captions for nearest neighbor search
with open(Config().TRAIN_CAPTIONS_PATH, 'r') as f:
    next(f)
    captions_doc = f.read()

mapping = defaultdict(list)
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], ' '.join(tokens[1:])
    image_id = image_id.split('.')[0]
    mapping[image_id].append(caption)

def clean_caption(caption):
    caption = caption.lower()
    caption = ''.join(c for c in caption if c.isalpha() or c.isspace())
    caption = ' '.join(word for word in caption.split() if len(word) > 1)
    return caption

all_captions = [clean_caption(caption) for captions in mapping.values() for caption in captions]
all_caption_embeddings = sentence_transformer.encode(all_captions, convert_to_tensor=True).cpu()

feature_extractor = VGG16(device).model
feature_extractor.eval()

if __name__ == "__main__":
    img_path = "/home/karan-chauhan/WorkStation/Project/Image-Caption-Generator/docs/Images/10815824_2997e03d76.jpg"
    caption = generate_caption(img_path, feature_extractor, all_captions, all_caption_embeddings)
    print(f"Generated caption: {caption}")
