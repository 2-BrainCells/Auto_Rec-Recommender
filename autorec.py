import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class AutoRec(nn.Module):
    """
    AutoRec: An autoencoder-based collaborative filtering model for recommendation systems.

    This model learns latent representations of users or items by reconstructing the input 
    interaction vector through a bottleneck architecture. The encoder compresses the input 
    into a low-dimensional representation, which the decoder then uses to reconstruct the 
    original input. AutoRec is particularly effective for scenarios with sparse rating data.

    Args:
        num_hidden (int): Number of hidden units in the bottleneck (latent) layer.
        num_items (int): Dimensionality of the input layer, typically equal to the number of items.
        dropout (float, optional): Dropout rate applied after the activation of the encoder. Defaults to 0.2.
    """
    def __init__(self, num_hidden, num_items, dropout=0.2):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_items, num_hidden)
        self.decoder = nn.Linear(num_hidden, num_items)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Defines the forward pass of the AutoRec model.

        Applies a sigmoid activation after the encoder transformation, followed by dropout for regularization.
        The decoder then reconstructs the input interaction vector from the hidden representation.

        Args:
            input (torch.Tensor): A tensor representing the input interaction vector for a user or item.

        Returns:
            torch.Tensor: The reconstructed interaction vector, representing predicted scores for all items.
        """
        hidden = self.dropout(self.sigmoid(self.encoder(input)))
        pred = self.decoder(hidden)
        # pred = torch.clamp(pred, min=0.0, max=1.0)
        # pred = 5 * torch.sigmoid(pred)
        
        return pred
    
class ARDataset(Dataset):
    """
    PyTorch Dataset for the AutoRec model, based on a user-item interaction matrix.

    This dataset wraps the interaction matrix used for training the AutoRec model. It includes a binary mask 
    that indicates observed (non-zero) entries in the interaction matrix, which is essential for loss calculation 
    during training.

    Args:
        interaction_matrix (np.ndarray or array-like): A 2D matrix where rows represent users and columns represent items. 
                                                       Entries contain interaction values (e.g., ratings or implicit feedback).

    Attributes:
        data (torch.Tensor): Tensor version of the interaction matrix, cast to float32.
        mask (torch.Tensor): A binary tensor of the same shape as `data`, where 1 indicates observed entries 
                             and 0 indicates missing/unobserved entries.
    """
    def __init__(self, interaction_matrix):
        self.data = torch.tensor(interaction_matrix, dtype=torch.float32)
        self.mask = (~(self.data <= 0)).float()
        
    def __len__(self):
        """
        Returns the number of user interaction vectors in the dataset.

        Returns:
            int: Number of rows in the interaction matrix.
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Retrieves the interaction vector and corresponding mask for a given user index.

        Args:
            idx (int): Index of the user.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The user's interaction vector.
                - The corresponding binary mask vector indicating observed entries.
        """
        return self.data[idx], self.mask[idx]