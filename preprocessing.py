import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

def read_data():
    """
    Loads and preprocesses user-item interaction data from the file 'values.csv'.

    This function executes the following steps:
    1. Reads the CSV file and selects columns starting from the 13th column onward.
    2. Determines the total number of users (rows) and items (columns) prior to any filtering.
    3. Replaces placeholder values indicating missing data ('NC', 'NSU', and whitespace) with NaN.
    4. Converts all entries to numeric values, coercing any invalid values to NaN.
    5. Filters out users who have more than 40% missing item ratings.
    6. Transforms the dataset into a long-form structurewith columns: 'User', 'Item', and 'Score'.
    7. Applies label encoding to item identifiers while preserving the original labels.
    8. Normalizes the score values to a [0, 1] scale by dividing by the maximum score (5.0).
    9. Displays the count of missing values remaining in the dataset.

    Returns:
        df (pd.DataFrame): A cleaned and normalized long-format DataFrame containing 
                           'User', 'Item', and 'Score' columns.
        num_users (int): The total number of users before filtering.
        num_items (int): The total number of items before filtering.
        """
    df = pd.read_csv('data\\values.csv')
    df = df.iloc[:, 12:]

    df = df.replace(['NC', 'NSU', ' '], torch.nan)
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
    df.shape

    df['Score'] /= 5.0 
    return df, num_users, num_items

def split_data(data, test_ratio=0.1): 
    """
    Splits the given dataset into training and testing subsets.

    This function randomly partitions the input DataFrame into training and testing sets 
    based on the specified test ratio. A fixed random seed is used to ensure reproducibility.

    Args:
        data (pd.DataFrame): The dataset to be split, typically containing user-item interactions.
        test_ratio (float, optional): Proportion of the data to be allocated to the test set. 
                                      Defaults to 0.1 (i.e., 10% test data).

    Returns:
        train_data (pd.DataFrame): The subset of the data used for training.
        test_data (pd.DataFrame): The subset of the data reserved for testing.
    """
    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=42) # data splitting

    return train_data, test_data

def load_data(data, num_users, num_items):
    """
    Converts a user-item interaction DataFrame into structured components for model input.

    This function extracts user indices, item indices, and corresponding scores from the input data,
    and constructs a user-item interaction matrix where each cell (i, j) represents the score given by user i to item j.

    Args:
        data (pd.DataFrame): A DataFrame containing user-item interactions, 
                             typically with columns [User, Item, Score].
        num_users (int): Total number of unique users.
        num_items (int): Total number of unique items.

    Returns:
        users (List[int]): A list of user indices corresponding to each interaction.
        items (List[int]): A list of item indices corresponding to each interaction.
        scores (List[float]): A list of interaction scores (e.g., ratings).
        inter (np.ndarray): A 2D NumPy array (interaction matrix) of shape (num_users, num_items) 
                            containing the user-item scores.
    """
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