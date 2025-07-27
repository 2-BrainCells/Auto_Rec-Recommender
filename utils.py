import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from preprocessing import device
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_value):
        if self.best_score is None or current_value < self.best_score - self.min_delta:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class OptunaEarlyStoppingCallback:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        current_best = study.best_value
        if current_best < self.best_value - self.min_delta:
            self.best_value = current_best
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            print(f"[OptunaEarlyStoppingCallback] Early stopping triggered after {self.patience} trials.")
            study.stop()


def masked_loss(predictions, targets, mask, loss_fn=nn.MSELoss(reduction='none')):
    """
    Compute masked loss between predicted and target values for sparse matrices.
    Only considers non-zero entries in the mask for loss calculation.
    """
    loss = loss_fn(predictions, targets) * mask
    return loss.sum() / mask.sum()

# def evaluator(net, test_data, inter_matrix, loss_fn, device=device):
#     """
#     Evaluate trained AutoRec model performance on test data.
#     Computes test loss and RMSE metrics for model validation.
#     """
#     net.eval()
#     scores = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for values, mask in test_data:
#             values, mask = values.to(device), mask.to(device)
#             preds = net(values)
#             scores.append(preds.to('cpu').numpy())
#             loss = masked_loss(preds, values, mask, loss_fn)
#             total_loss += loss.item()

#     recons = np.vstack(scores)
#     inter_matrix = np.array(inter_matrix, dtype=np.float16)
#     rmse = np.sqrt(np.sum(np.square(inter_matrix - np.sign(inter_matrix) * recons)) / np.sum(np.sign(inter_matrix)))
#     loss = total_loss / len(test_data)

#     return loss, rmse

def evaluator(net, test_data, loss_fn, device=device):
    """
    Evaluate trained AutoRec model performance on test data.
    Computes test loss and RMSE metrics for model validation.
    Both metrics are calculated over the same observed (masked) ratings.
    """
    net.eval()
    all_preds = []
    all_trues = []
    all_masks = []
    total_loss = 0.0

    with torch.no_grad():
        for values, mask in test_data:
            values, mask = values.to(device), mask.to(device)
            preds = net(values)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(values.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            loss = masked_loss(preds, values, mask, loss_fn)
            total_loss += loss.item()

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Compute masked RMSE over all observed entries
    mse = np.sum(((preds - trues) ** 2) * masks) / np.sum(masks)
    rmse = np.sqrt(mse)

    avg_loss = total_loss / len(test_data)
    return avg_loss, rmse


def train_ranking(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, device=None, evaluator=None, inter_mat=None, early_stopping_patience=5):
    """
    Train AutoRec model using masked loss with evaluation tracking.
    Performs complete training loop with loss monitoring and evaluation.
    """
    net.train()
    train_loss, test_loss, test_rmse = [], [], []
    early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=1e-4)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch, mask in train_iter:
            batch, mask = batch.to(device), mask.to(device)
            optimizer.zero_grad()
            predictions = net(batch)
            loss = masked_loss(predictions, batch, mask, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_l = total_loss / len(train_iter)

        if evaluator:
            test_l, rmse = evaluator(net, test_iter, loss_fn, device)
        else:
            test_l, rmse = None, None

        train_loss.append(train_l)
        test_loss.append(test_l)
        test_rmse.append(rmse)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_l:.3f}, Test Loss: {test_l:.3f}, Test RMSE: {rmse:.3f}")
        
        if rmse is not None:
            early_stopper(rmse)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return train_loss, test_loss, test_rmse

def load_and_use_config(config_path="Auto_Rec_best_params"):
    """
    Load YAML configuration file containing model hyperparameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class AutoRecPredictor:
    """
    Prediction wrapper for trained AutoRec model.
    Provides methods for generating recommendations for both existing and new users.
    """
    
    def __init__(self, model, interaction_matrix, device):
        """
        Initialize predictor with trained model and interaction data.
        """
        self.model = model
        self.interaction_matrix = interaction_matrix
        self.device = device
        self.model.eval()

    def predict_existing_user(self, user_id, top_k=10):
        """
        Generate recommendations for existing user using AutoRec reconstruction.
        """
        user_vector = torch.tensor(self.interaction_matrix[user_id],
                                   dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(user_vector).squeeze().cpu().numpy()

        unrated_items = np.where(self.interaction_matrix[user_id] == 0)[0]
        unrated_predictions = predictions[unrated_items]
        top_indices = np.argsort(unrated_predictions)[-top_k:][::-1]
        recommended_items = unrated_items[top_indices]
        predicted_scores = unrated_predictions[top_indices]

        return list(zip(recommended_items, predicted_scores))

    def predict_new_user_popularity(self, top_k=10):
        """
        Generate popularity-based recommendations for new users.
        """
        item_popularity = np.sum(self.interaction_matrix > 0, axis=0)
        popular_indices = np.argsort(item_popularity)[-top_k:][::-1]
        popular_scores = item_popularity[popular_indices]

        if np.max(popular_scores) > 0:
            popular_scores = popular_scores / np.max(popular_scores)

        return list(zip(popular_indices, popular_scores))

    def predict_new_user_average(self, top_k=10):
        """
        Generate recommendations for new users using average user profile.
        """
        avg_user_profile = np.mean(self.interaction_matrix, axis=0)
        avg_user_tensor = torch.tensor(avg_user_profile, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(avg_user_tensor).squeeze().cpu().numpy()

        top_indices = np.argsort(predictions)[-top_k:][::-1]
        predicted_scores = predictions[top_indices]

        return list(zip(top_indices, predicted_scores))

def generate_autorec_recommendations(model, interaction_matrix, device,
                                     user_id=None, new_user_method='popularity', top_k=10):
    """
    Generate recommendations using AutoRec model for existing or new users.
    """
    predictor = AutoRecPredictor(model, interaction_matrix, device)

    if user_id is not None:
        return predictor.predict_existing_user(user_id, top_k)
    else:
        if new_user_method == 'popularity':
            return predictor.predict_new_user_popularity(top_k)
        elif new_user_method == 'average':
            return predictor.predict_new_user_average(top_k)
        else:
            raise ValueError("new_user_method must be 'popularity' or 'average'")

def display_recommendations(recommendations, title="Recommendations"):
    """
    Display formatted recommendation results.
    """
    print(f"\n=== {title} ===")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"{i}. Item {item_id}: {score:.4f}")

class PreferenceBasedPredictor:
    """
    Advanced predictor for new users based on preference slider inputs.
    Combines collaborative filtering with preference-based similarity matching.
    """
    
    def __init__(self, model, interaction_matrix, device):
        """
        Initialize preference-based predictor with model and data.
        """
        self.model = model
        self.interaction_matrix = interaction_matrix
        self.device = device
        self.model.eval()

    def create_preference_profile(self, preferences, normalize=True):
        """
        Convert user preference sliders to interaction profile vector.
        """
        profile = np.zeros(self.interaction_matrix.shape[1])

        for category, data in preferences.items():
            rating = data['rating']
            item_ids = data['item_ids']
            normalized_rating = (rating - 1) / 4.0 if normalize else rating / 5.0

            for item_id in item_ids:
                if item_id < len(profile):
                    profile[item_id] = normalized_rating

        return profile

    def find_similar_users(self, preference_profile, top_k=10, min_similarity=0.1):
        """
        Find users with similar preferences using cosine similarity.
        """
        similarities = cosine_similarity([preference_profile], self.interaction_matrix)[0]
        valid_users = np.where(similarities >= min_similarity)[0]

        if len(valid_users) == 0:
            return [], []

        top_user_indices = valid_users[np.argsort(similarities[valid_users])[-top_k:]]
        top_similarities = similarities[top_user_indices]

        return top_user_indices, top_similarities

    def predict_with_preferences(self, preferences, top_k=10, hybrid_weight=0.7):
        """
        Generate recommendations using preference-based similarity and AutoRec.
        """
        preference_profile = self.create_preference_profile(preferences)
        similar_users, similarities = self.find_similar_users(preference_profile, top_k=10)

        if len(similar_users) == 0:
            return self.fallback_popularity_recommendations(top_k)

        collaborative_scores = self.generate_collaborative_scores(
            similar_users, similarities, preference_profile
        )

        autorec_scores = self.generate_autorec_scores(preference_profile)

        final_scores = self.combine_scores(
            collaborative_scores, autorec_scores,
            weight=hybrid_weight, preference_profile=preference_profile
        )

        top_items = np.argsort(final_scores)[-top_k:][::-1]
        top_scores = final_scores[top_items]

        return list(zip(top_items, top_scores))

    def generate_collaborative_scores(self, similar_users, similarities, preference_profile):
        """
        Generate scores based on similar users' preferences and interactions.
        """
        scores = np.zeros(len(preference_profile))
        
        for user_idx, similarity in zip(similar_users, similarities):
            user_ratings = self.interaction_matrix[user_idx]
            for item_id, rating in enumerate(user_ratings):
                if rating > 0 and preference_profile[item_id] == 0:
                    scores[item_id] += similarity * rating

        if np.max(scores) > 0:
            scores = scores / np.max(scores)

        return scores

    def generate_autorec_scores(self, preference_profile):
        """
        Generate scores using AutoRec with synthetic preference profile.
        """
        profile_tensor = torch.tensor(preference_profile, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(profile_tensor).squeeze().cpu().numpy()

        if np.max(predictions) > 0:
            predictions = predictions / np.max(predictions)

        return predictions

    def combine_scores(self, collaborative_scores, autorec_scores, weight=0.7, preference_profile=None):
        """
        Combine collaborative filtering scores with AutoRec scores using weighted average.
        """
        combined = weight * collaborative_scores + (1 - weight) * autorec_scores

        if preference_profile is not None:
            preference_boost = preference_profile * 0.3
            combined = combined + preference_boost

        return combined

    def fallback_popularity_recommendations(self, top_k):
        """
        Generate popularity-based recommendations when no similar users are found.
        """
        item_popularity = np.sum(self.interaction_matrix > 0, axis=0)
        popular_indices = np.argsort(item_popularity)[-top_k:][::-1]
        popular_scores = item_popularity[popular_indices]

        if np.max(popular_scores) > 0:
            popular_scores = popular_scores / np.max(popular_scores)

        return list(zip(popular_indices, popular_scores))

def generate_preference_based_recommendations(model, interaction_matrix, device,
                                              preferences, top_k=10):
    """
    Generate recommendations for new users based on preference slider inputs.
    """
    predictor = PreferenceBasedPredictor(model, interaction_matrix, device)
    return predictor.predict_with_preferences(preferences, top_k)

def plot_metrics(history, title, figsize=(12, 4)):
    """
    Plot training and validation metrics for model performance visualization.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].plot(history['loss'], label='Train Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    for metric, values in history.items():
        if metric not in ['loss', 'val_loss']:
            ax[1].plot(values, label=metric)

    ax[1].set_title('Metrics')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')
    ax[1].legend()

    plt.suptitle(title)
    plt.show()

def calculate_recommendation_diversity(recommendations, interaction_matrix):
    """
    Calculate diversity metrics for recommendation quality assessment.
    """
    if len(recommendations) < 2:
        return 0.0

    item_ids = [item_id for item_id, _ in recommendations]
    item_vectors = interaction_matrix[:, item_ids].T
    
    similarities = cosine_similarity(item_vectors)
    diversity = 1 - np.mean(similarities[np.triu_indices_from(similarities, k=1)])
    
    return max(0.0, diversity)

def evaluate_recommendation_coverage(recommendations, total_items):
    """
    Calculate coverage percentage of recommendations relative to total items.
    """
    recommended_items = set([item_id for item_id, _ in recommendations])
    coverage = len(recommended_items) / total_items
    return coverage

def compute_recommendation_novelty(recommendations, interaction_matrix):
    """
    Calculate novelty score based on item popularity in the dataset.
    """
    item_popularity = np.sum(interaction_matrix > 0, axis=0)
    total_interactions = np.sum(item_popularity)
    
    if total_interactions == 0:
        return 0.0
    
    novelty_scores = []
    for item_id, _ in recommendations:
        if item_id < len(item_popularity):
            popularity_ratio = item_popularity[item_id] / total_interactions
            novelty = -np.log2(popularity_ratio + 1e-10)
            novelty_scores.append(novelty)
    
    return np.mean(novelty_scores) if novelty_scores else 0.0
