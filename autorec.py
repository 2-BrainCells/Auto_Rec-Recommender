import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class AutoRec(nn.Module):
    """
    AutoRec autoencoder-based collaborative filtering model for recommendation systems.
    Implements encoder-decoder architecture to learn user-item interaction patterns.
    """
    
    def __init__(self, num_hidden, num_items, dropout=0.2):
        """
        Initialize AutoRec model with specified architecture parameters.
        
        Args:
            num_hidden: Number of hidden units in the encoder/decoder layers
            num_items: Total number of items in the dataset
            dropout: Dropout rate for regularization
        """
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_items, num_hidden)
        self.decoder = nn.Linear(num_hidden, num_items)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Forward pass through the AutoRec model.
        Encodes input through hidden layer and decodes back to original dimensions.
        """
        hidden = self.dropout(self.sigmoid(self.encoder(input)))
        pred = self.decoder(hidden)
        return pred

class ARDataset(Dataset):
    """
    PyTorch Dataset class for AutoRec model training and evaluation.
    Handles user-item interaction matrix data with masking for missing values.
    """
    
    def __init__(self, interaction_matrix):
        """
        Initialize dataset with interaction matrix and create corresponding mask.
        
        Args:
            interaction_matrix: User-item interaction matrix with ratings
        """
        self.data = torch.tensor(interaction_matrix, dtype=torch.float32)
        self.mask = (~(self.data <= 0)).float()

    def __len__(self):
        """Return the number of users (rows) in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Get a single user's interaction vector and corresponding mask.
        
        Args:
            idx: User index
            
        Returns:
            Tuple of (user_interactions, mask)
        """
        return self.data[idx], self.mask[idx]
