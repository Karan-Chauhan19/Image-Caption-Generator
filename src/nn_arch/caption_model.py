"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

import torch
import torch.nn as nn
from nn_arch.attention import Attention

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
        encoder_outputs = self.encoder_relu(self.encoder_fc(self.encoder_dropout(img_features)))
        encoder_outputs = encoder_outputs.unsqueeze(1)
        
        embedded = self.embedding(captions)
        
        batch_size = img_features.size(0)
        hidden = torch.zeros(1, batch_size, 512).to(img_features.device)
        cell = torch.zeros(1, batch_size, 512).to(img_features.device)
        
        outputs = []
        for t in range(captions.size(1)):
            context, _ = self.attention(hidden.squeeze(0), encoder_outputs)
            lstm_input = torch.cat((context, embedded[:, t, :].unsqueeze(1)), dim=2)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            output = self.dropout(output)
            output = self.fc(output.squeeze(1))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)