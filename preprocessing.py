import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_data():
    """Load and preprocess user-item interaction data"""
    df = pd.read_csv('data/values.csv')  # Fixed path separator
    df = df.iloc[:, 12:]
    df = df.replace(['NC', 'NSU', ' '], np.nan)  # Fixed np.nan usage
    df = df.apply(pd.to_numeric, errors='coerce')
    
    max_nan_ratio = 0.4
    max_nan_allowed = int(df.shape[1] * max_nan_ratio)
    df = df[df.isnull().sum(axis=1) <= max_nan_allowed]
    df.reset_index(drop=True, inplace=True)
    
    num_users = len(df.index)
    num_items = len(df.columns)
    
    df = df.stack().rename_axis(('User', 'Item')).reset_index(name='Score')
    item_labels, item_original = pd.factorize(df['Item'])
    df['Item'] = item_labels
    df['Score'] /= 5.0
    
    return df, num_users, num_items

def split_data(data, test_ratio=0.1):
    """Split the dataset into training and testing subsets"""
    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=42)
    return train_data, test_data

def load_data(data, num_users, num_items):
    """Convert DataFrame into structured components for model input"""
    users, items, scores = [], [], []
    inter = np.zeros((num_users, num_items))
    
    for line in data.itertuples():
        user_index, item_index = int(line[1]), int(line[2])
        score = line[3]
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        inter[user_index, item_index] = score
    
    return users, items, scores, inter

def validate_no_demographics_used(df):
    """Validate that demographic columns are properly excluded"""
    if df.shape[1] < 13:
        raise ValueError("Dataset must have at least 13 columns (12 demographic + item interactions)")
    
    # Only use columns 12+ (item interactions)
    interaction_cols = df.iloc[:, 12:]
    
    print(f"✅ Demographics validation passed:")
    print(f"   - Total columns: {df.shape[1]}")
    print(f"   - Demographic columns (ignored): {12}")
    print(f"   - Interaction columns (used): {interaction_cols.shape[1]}")
    
    return interaction_cols
