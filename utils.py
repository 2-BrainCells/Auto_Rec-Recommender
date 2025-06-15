import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from preprocessing import device
import matplotlib.pyplot as plt

def masked_loss(predictions, targets, mask, loss_fn=nn.MSELoss(reduction='none')):
    """
    Computes the masked loss between predicted and target values, considering only observed entries.

    This function calculates the element-wise loss (default: mean squared error) and applies a binary mask 
    to ignore unobserved interactions (e.g., missing or zero values). The final loss is normalized by 
    the number of observed entries to ensure consistent scaling.

    Args:
        predictions (torch.Tensor): The predicted user-item interaction scores.
        targets (torch.Tensor): The ground truth interaction scores.
        mask (torch.Tensor): A binary tensor of the same shape as `predictions` indicating observed entries (1) and missing ones (0).
        loss_fn (Callable, optional): Loss function to apply element-wise. Defaults to `nn.MSELoss(reduction='none')`.

    Returns:
        torch.Tensor: The scalar masked loss value, averaged over the observed entries.
    """
    loss = loss_fn(predictions, targets) * mask  # Ignore unobserved values
    return loss.sum() / mask.sum()

def evaluator(net, test_data, inter_matrix, loss_fn, device=device):
    """
    Evaluates the performance of a trained AutoRec model on a test dataset.

    This function performs inference on the test data using the provided model and computes:
    1. The average masked loss over the test set.
    2. The root mean squared error (RMSE) between the reconstructed and original interaction matrix 
       over the observed entries.

    Args:
        net (nn.Module): The trained AutoRec model to be evaluated.
        test_data (DataLoader): A PyTorch DataLoader providing batches of test data in the form of (input, mask) pairs.
        inter_matrix (array-like): The original interaction matrix used for computing RMSE.
        loss_fn (Callable): The loss function used to compute reconstruction error (e.g., masked MSE).
        device (torch.device): The computation device (CPU or GPU) on which the model operates.

    Returns:
        loss (float): The average masked loss across the test dataset.
        rmse (float): The root mean squared error between the predicted and actual interactions 
                      over the observed (non-zero) entries.
    """
    net.eval()
    scores = []
    total_loss = 0.0
    
    with torch.no_grad():
        for values, mask in test_data:
            values, mask = values.to(device), mask.to(device)
            preds = net(values)
            # preds = torch.clamp(preds, min=0, max=5)
            scores.append(preds.to('cpu').numpy())
            loss = masked_loss(preds, values, mask, loss_fn) 
            total_loss += loss.item()

    recons = np.vstack(scores)

    inter_matrix = np.array(inter_matrix, dtype=np.float16)
    rmse = np.sqrt(np.sum(np.square(inter_matrix - np.sign(inter_matrix) * recons)) / np.sum(np.sign(inter_matrix)))
    loss = total_loss / len(test_data)

    return loss, rmse

def train_ranking(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, device=None, evaluator=None, inter_mat=None):
    """
    Trains the AutoRec model using masked loss and evaluates its performance over multiple epochs.

    This function handles the full training loop for the AutoRec model, including:
    - Forward and backward passes for each batch.
    - Masked loss computation to account only for observed ratings.
    - Parameter updates using the specified optimizer.
    - Periodic evaluation using an external evaluator function, if provided.

    Args:
        net (nn.Module): The AutoRec model to be trained.
        train_iter (DataLoader): DataLoader providing the training data in (input, mask) batches.
        test_iter (DataLoader): DataLoader providing the testing data for evaluation.
        loss_fn (Callable): Loss function (e.g., MSE) used to compute training loss.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        num_epochs (int): Number of training epochs.
        device (torch.device, optional): Device to run the training on (e.g., 'cuda' or 'cpu'). Defaults to None.
        evaluator (Callable, optional): Function to evaluate model performance on the test set. 
                                        Should return (test_loss, rmse). Defaults to None.
        inter_mat (array-like, optional): The interaction matrix used for RMSE calculation during evaluation. Required if `evaluator` is used.

    Returns:
        train_loss (List[float]): Training loss recorded at each epoch.
        test_loss (List[float]): Test loss recorded at each epoch (if evaluator is provided; otherwise, contains None).
        test_rmse (List[float]): Test RMSE recorded at each epoch (if evaluator is provided; otherwise, contains None).
    """
    net.train()  # Set model to training mode
    # Setting up plot for visualization
    train_loss, test_loss, test_rmse = [], [], []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch, mask in train_iter:
            batch, mask = batch.to(device), mask.to(device)

            optimizer.zero_grad()  # Reset gradients
            predictions = net(batch) # Forward pass
            loss = masked_loss(predictions, batch, mask, loss_fn)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            total_loss += loss.item()

        train_l = total_loss / len(train_iter)

        # Evaluate on test set
        if evaluator:
            test_l, rmse = evaluator(net, test_iter, inter_mat, loss_fn)
        else:
            test_l, rmse = None # No evaluation function provided

        # Compute epoch loss
        train_loss.append(train_l)
        test_loss.append(test_l)
        test_rmse.append(rmse)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_l:.3f}, Test Loss: {test_l:.3f}, Test RMSE: {rmse:.3f}")

    return train_loss, test_loss, test_rmse

def calculate_ranking_metrics(predictions, ground_truth, masks, k=5):
    """
    Calculate ranking metrics for top-k recommendations
    
    Args:
        predictions: Predicted ratings matrix (users x items)
        ground_truth: Binary matrix where 1 indicates relevant items
        k: Number of items to recommend
    
    Returns:
        Dictionary containing Precision@K, Recall@K, and NDCG@K
    """
    num_users = predictions.shape[0]
    precision_k = np.zeros(num_users)
    recall_k = np.zeros(num_users)
    ndcg_k = np.zeros(num_users)
    
    for user_idx in range(num_users):
        # Get user predictions and ground truth
        user_preds = predictions[user_idx]
        user_truth = ground_truth[user_idx]
        
        # Get indices of top-k recommended items
        if np.sum(masks[user_idx]) > k:
            candidate_items = np.where(masks[user_idx])[0]
            candidate_preds = user_preds[candidate_items]
            recommended_items_idx = candidate_items[np.argsort(-candidate_preds)[:k]]
        else:
            # If fewer than k unrated items, take all
            recommended_items_idx = np.argsort(-user_preds)[:k]
        
        # Calculate precision@k
        relevant_and_recommended = np.sum(user_truth[recommended_items_idx] > 0)
        precision_k[user_idx] = relevant_and_recommended / k
        
        # Calculate recall@k
        total_relevant = np.sum(user_truth > 0)
        if total_relevant > 0:
            recall_k[user_idx] = relevant_and_recommended / total_relevant
        
        # Calculate NDCG@k
        dcg_k = 0
        idcg_k = 0
        
        # Create ideal ranking (all relevant items first)
        ideal_ranking = np.argsort(-user_truth)[:k]
        
        # Calculate DCG
        for i, item_idx in enumerate(recommended_items_idx):
            if user_truth[item_idx] > 0:
                # Using log2(i+2) to handle i=0 case
                dcg_k += 1 / np.log2(i + 2)
        
        # Calculate IDCG
        for i, item_idx in enumerate(ideal_ranking):
            if user_truth[item_idx] > 0:
                idcg_k += 1 / np.log2(i + 2)
        
        # Calculate NDCG
        if idcg_k > 0:
            ndcg_k[user_idx] = dcg_k / idcg_k
    
    
    print(f"Precision@{k}: {np.mean(precision_k)}, Recall@{k}: {np.mean(recall_k)}, NDCG@{k}: {np.mean(ndcg_k)}")

def calculate_coverage_diversity(predictions, item_metadata=None, k=5):
    """
    Calculate coverage and diversity metrics
    
    Args:
        predictions: Predicted ratings matrix (users x items)
        item_metadata: Optional metadata for calculating content diversity
        k: Number of items to recommend
    
    Returns:
        Dictionary containing coverage and diversity metrics
    """
    num_users, num_items = predictions.shape
    
    # Get top-k recommendations for each user
    top_k_items = np.zeros((num_users, k), dtype=int)
    for user_idx in range(num_users):
        top_k_items[user_idx] = np.argsort(-predictions[user_idx])[:k]
    
    # Calculate catalog coverage
    recommended_items = np.unique(top_k_items.flatten())
    catalog_coverage = len(recommended_items) / num_items
    
    # Calculate recommendation diversity (Intra-List Similarity)
    if item_metadata is not None:
        # Calculate average pairwise similarity within recommendation lists
        diversity = 0
        for user_idx in range(num_users):
            user_recs = top_k_items[user_idx]
            pairwise_similarity = 0
            pair_count = 0
            
            for i in range(k):
                for j in range(i+1, k):
                    item_i = user_recs[i]
                    item_j = user_recs[j]
                    # Calculate cosine similarity between items using metadata
                    sim = np.dot(item_metadata[item_i], item_metadata[item_j]) / (
                        np.linalg.norm(item_metadata[item_i]) * np.linalg.norm(item_metadata[item_j])
                    )
                    pairwise_similarity += sim
                    pair_count += 1
            
            user_diversity = 1 - (pairwise_similarity / pair_count if pair_count > 0 else 0)
            diversity += user_diversity
        
        diversity /= num_users
    else:
        diversity = None
    
    print(f"Catalog_Coverage@{k}: {catalog_coverage}, Recommendation_Diversity@{k}: {diversity}")

def evaluate_recommendation_model(model, test_loader, item_features=None, ks=[5, 10, 20]):
    """
    Perform comprehensive evaluation of a recommendation model
    
    Args:
        model: Trained recommendation model
        test_loader: DataLoader with test data
        test_matrix: Complete test interaction matrix
        item_features: Optional item features for diversity calculation
        ks: List of k values for ranking metrics
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []
    ground_truth = []
    masks = []
    
    with torch.no_grad():
        for values, mask in test_loader:
            values = values.to(device)
            mask = mask.to(device)
            preds = model(values)
            
            predictions.append(preds.cpu().numpy())
            ground_truth.append(values.cpu().numpy())
            masks.append(mask.cpu().numpy())
    
    # Concatenate batch results
    # print(ground_truth, predictions)
    predictions = np.vstack(predictions)
    ground_truth = np.vstack(ground_truth)
    masks = np.vstack(masks)
    # print(ground_truth.shape, predictions.shape)
    
    # Prepare binary relevance matrix for ranking metrics
    # For example, consider items with rating >= 4 as relevant
    relevance_threshold = 0.6  # Assuming normalized ratings [0,1]
    binary_relevance = (ground_truth >= relevance_threshold).astype(int)
    
    # Calculate ranking metrics for different k values
    ranking_metrics = {}
    for k in ks:
        calculate_ranking_metrics(predictions, ground_truth, masks, k)
    
    # Calculate coverage and diversity
    coverage_diversity = {}
    for k in ks:
        (calculate_coverage_diversity(predictions, item_features, k))
        
def load_and_use_config(config_path="Auto_Rec_best_params"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def precision_recall_at_k(y_preds, y_true, X_test_users, k=10, threshold=0.6):
    """
    Calculate Precision@K and Recall@K for recommendation system
    
    Args:
        y_preds: Predicted scores (numpy array)
        y_true: True ratings (numpy array) 
        X_test_users: User features for test set to group by user
        k: Number of top recommendations to consider
        threshold: Rating threshold to consider an item as relevant (normalized, e.g., 0.6 for 3/5)
    
    Returns:
        precision_k, recall_k: Average precision and recall at k across all users
    """
    unique_users = np.unique(X_test_users, axis=0)
    
    precisions = []
    recalls = []
    
    for user in unique_users:
        # Get indices for this user's interactions
        user_indices = np.where((X_test_users == user).all(axis=1))[0]
        
        if len(user_indices) == 0:
            continue
            
        # Get predictions and true ratings for this user
        user_preds = y_preds[user_indices]
        user_true = y_true[user_indices]
        
        # Handle case where user has fewer than k items
        actual_k = min(k, len(user_preds))
        
        # Get top-k predictions (highest predicted scores)
        top_k_indices = np.argsort(user_preds)[-actual_k:][::-1]
        top_k_true = user_true[top_k_indices]
        
        # Calculate relevant items (ratings above threshold)
        relevant_in_top_k = np.sum(top_k_true >= threshold)
        total_relevant = np.sum(user_true >= threshold)
        
        # Calculate precision@k and recall@k
        precision = relevant_in_top_k / actual_k if actual_k > 0 else 0
        recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.mean(precisions), np.mean(recalls)

def plot_metrics(history: dict, title: str, figsize: tuple=(12, 4)) -> None:
        '''
        Plot the training and validation losses in one figure and the other metrics in another figure
        '''
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].plot(history['loss'], label='Train Loss')
        ax[0].plot(history['val_loss'], label='Validation Loss')
        ax[0].set_title('Training and Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Metrics plot
        for metric, values in history.items():
            if metric not in ['loss', 'val_loss']:
                ax[1].plot(values, label=metric)
            
        ax[1].set_title('Metrics')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Value')
        ax[1].legend()
        plt.suptitle(title)
        plt.show()

class AutoRecPredictor:
    """Simple predictor for AutoRec model - works only with interaction data"""
    
    def __init__(self, model, interaction_matrix, device):
        self.model = model
        self.interaction_matrix = interaction_matrix
        self.device = device
        self.model.eval()
    
    def predict_existing_user(self, user_id, top_k=10):
        """Predict items for existing user using AutoRec reconstruction"""
        user_vector = torch.tensor(self.interaction_matrix[user_id], 
                                 dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(user_vector).squeeze().cpu().numpy()
        
        # Filter out items user already rated
        unrated_items = np.where(self.interaction_matrix[user_id] == 0)[0]
        unrated_predictions = predictions[unrated_items]
        top_indices = np.argsort(unrated_predictions)[-top_k:][::-1]
        
        recommended_items = unrated_items[top_indices]
        predicted_scores = unrated_predictions[top_indices]
        
        return list(zip(recommended_items, predicted_scores))
    
    def predict_new_user_popularity(self, top_k=10):
        """For new users: recommend most popular items (no demographic data)"""
        # Calculate item popularity based on number of interactions
        item_popularity = np.sum(self.interaction_matrix > 0, axis=0)
        popular_indices = np.argsort(item_popularity)[-top_k:][::-1]
        popular_scores = item_popularity[popular_indices]
        
        # Normalize scores
        if np.max(popular_scores) > 0:
            popular_scores = popular_scores / np.max(popular_scores)
        
        return list(zip(popular_indices, popular_scores))
    
    def predict_new_user_average(self, top_k=10):
        """For new users: use average user profile to make predictions"""
        # Create average user profile from all users
        avg_user_profile = np.mean(self.interaction_matrix, axis=0)
        avg_user_tensor = torch.tensor(avg_user_profile, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(avg_user_tensor).squeeze().cpu().numpy()
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        predicted_scores = predictions[top_indices]
        
        return list(zip(top_indices, predicted_scores))

def generate_autorec_recommendations(model, interaction_matrix, device, 
                                   user_id=None, new_user_method='popularity', top_k=10):
    """
    Generate recommendations using only AutoRec capabilities
    
    Args:
        model: Trained AutoRec model
        interaction_matrix: User-item interaction matrix
        device: Computing device
        user_id: ID for existing user (None for new users)
        new_user_method: 'popularity' or 'average' for new users
        top_k: Number of recommendations
    """
    predictor = AutoRecPredictor(model, interaction_matrix, device)
    
    if user_id is not None:
        # Existing user
        return predictor.predict_existing_user(user_id, top_k)
    else:
        # New user - simple approaches since AutoRec has no preference handling
        if new_user_method == 'popularity':
            return predictor.predict_new_user_popularity(top_k)
        elif new_user_method == 'average':
            return predictor.predict_new_user_average(top_k)
        else:
            raise ValueError("new_user_method must be 'popularity' or 'average'")

def display_recommendations(recommendations, title="Recommendations"):
    """Display recommendations"""
    print(f"\n=== {title} ===")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"{i}. Item {item_id}: {score:.4f}")
