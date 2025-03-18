"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

import torch
import torch.nn as nn

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