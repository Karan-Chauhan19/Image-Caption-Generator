"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

import torch
import torch.nn as nn
from nn_arch.attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CaptionModel(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=512):
        super(CaptionModel, self).__init__()

        # Encoder
        self.encoder_fc = nn.Linear(4096, hidden_dim)
        self.encoder_dropout = nn.Dropout(0.5)
        self.encoder_relu = nn.ReLU()

        # Decoder with Attention
        self.embedding_fc = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, embedding_dim)  # Output sentence transformer embedding dim

    def forward(self, img_features, caption_embeddings):
        # img_features: (batch_size, 4096)
        # caption_embeddings: (batch_size, embedding_dim)

        # Encode image features
        encoder_outputs = self.encoder_relu(self.encoder_fc(self.encoder_dropout(img_features)))
        encoder_outputs = encoder_outputs.unsqueeze(1)

        # Process caption embeddings
        embedded = self.embedding_fc(caption_embeddings).unsqueeze(1)

        # Initialize LSTM state
        batch_size = img_features.size(0)
        hidden = torch.zeros(1, batch_size, 512).to(device)
        cell = torch.zeros(1, batch_size, 512).to(device)

        # Apply attention and LSTM
        context, _ = self.attention(hidden.squeeze(0), encoder_outputs)
        lstm_input = torch.cat((context, embedded), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.dropout(output)
        output = self.fc(output.squeeze(1))

        return output