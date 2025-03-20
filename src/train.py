"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J Unniversity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from nn_arch.vgg16 import VGG16
from config.config import Config
from dataset import CaptionDataset
from gpu_config.check import check_gpu
from torch.utils.data import DataLoader
from utils.embeddings import EmbeddingHandler
from nn_arch.caption_model import CaptionModel

def main():
    check_gpu()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embedding = EmbeddingHandler()
    vgg = VGG16()

    mapping = EmbeddingHandler().process_captions(Config().TRAIN_CAPTIONS_PATH)
    features = vgg.extract_features(Config().TRAIN_IMAGE_PATH)
    caption_embeddings = embedding.generate_embeddings(mapping)
    max_length = embedding.get_max_length(mapping)
    image_ids = [img_id for img_id in mapping.keys() if img_id in features]
    split = int(len(image_ids) * 0.95)
    train = image_ids[:split]

    #Training
    batch_size = 16
    train_dataset = CaptionDataset(train, mapping, features, caption_embeddings, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CaptionModel(embedding_dim=384).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_idx, (img_features, caption_embeddings) in enumerate(train_loader):
            img_features = img_features.to(device)
            caption_embeddings = caption_embeddings.to(device)

            optimizer.zero_grad()
            outputs = model(img_features, caption_embeddings)

            loss = criterion(outputs, caption_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), Config().PATH_TO_SAVE_TRAINED_MODEL)
