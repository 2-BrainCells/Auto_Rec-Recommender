import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class AutoRec(nn.Module):
    """
    AutoRec: An autoencoder-based collaborative filtering model for recommendation systems.
    """

    def __init__(self, num_hidden, num_items, dropout=0.2):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_items, num_hidden)
        self.decoder = nn.Linear(num_hidden, num_items)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """Forward pass of the AutoRec model."""
        hidden = self.dropout(self.sigmoid(self.encoder(input)))
        pred = self.decoder(hidden)
        return pred

class ARDataset(Dataset):
    """PyTorch Dataset for the AutoRec model."""

    def __init__(self, interaction_matrix):
        self.data = torch.tensor(interaction_matrix, dtype=torch.float32)
        self.mask = (~(self.data <= 0)).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]
