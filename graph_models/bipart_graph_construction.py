import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def preprocess_customer_ids(df, df_customer_features):
    """
    Ensure CustomerID columns and indices are strings and consistent in both dataframes.
    
    Args:
        df (pd.DataFrame): Original transaction dataframe.
        df_customer_features (pd.DataFrame): Customer feature dataframe with CustomerID as index.
        
    Returns:
        Tuple: (df_filtered, df_customer_features_processed)
    """
    df = df.copy()
    df_customer_features = df_customer_features.copy()

    # Convert CustomerID to string for consistency
    df['CustomerID'] = df['CustomerID'].astype(str)
    df_customer_features.index = df_customer_features.index.astype(str)

    # Filter df to keep only customers in features
    df_filtered = df[df['CustomerID'].isin(df_customer_features.index)]

    return df_filtered, df_customer_features

def create_index_mappings(df, df_customer_features):
    """
    Create user and item index mappings for graph construction.
    
    Args:
        df (pd.DataFrame): Filtered transaction data.
        df_customer_features (pd.DataFrame): Customer features with CustomerID as index.
        
    Returns:
        Tuple: (user2idx, item2idx, valid_user_ids, item_ids)
    """
    # Ensure CustomerID is string and consistent
    df['CustomerID'] = df['CustomerID'].astype(float).astype(int).astype(str)

    user_ids = df['CustomerID'].unique()
    item_ids = df['StockCode'].unique()

    # Keep only user_ids present in customer features
    valid_user_ids = [uid for uid in user_ids if uid in df_customer_features.index]

    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    item2idx = {iid: i + len(user_ids) for i, iid in enumerate(item_ids)}  # item indices start after user indices

    return user2idx, item2idx, valid_user_ids, item_ids

def build_edge_index(df, user2idx, item2idx):
    """
    Build edge index tensor for PyG from user-item interactions.
    
    Args:
        df (pd.DataFrame): Filtered transaction data.
        user2idx (dict): Mapping from user IDs to indices.
        item2idx (dict): Mapping from item IDs to indices.
        
    Returns:
        torch.LongTensor: edge_index of shape [2, num_edges]
    """
    edge_index = []
    for _, row in df.iterrows():
        user_id = row['CustomerID']
        item_id = row['StockCode']
        if pd.notna(user_id) and item_id in item2idx:
            user_idx = user2idx[user_id]
            item_idx = item2idx[item_id]
            edge_index.append([user_idx, item_idx])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def build_node_features(df_customer_features, user_ids, item_ids):
    """
    Create node feature matrix for users and items.
    
    Args:
        df_customer_features (pd.DataFrame): Customer feature dataframe.
        user_ids (list): List of user IDs in order.
        item_ids (list): List of item IDs.
        
    Returns:
        torch.FloatTensor: Combined node feature matrix
    """
    # Align customer features by user_ids order
    df_customer_features = df_customer_features.loc[user_ids]

    # Convert features to float32 tensor
    df_customer_features = df_customer_features.astype('float32')
    x_user = torch.tensor(df_customer_features.values, dtype=torch.float)

    # Placeholder item features as zeros
    x_item = torch.zeros((len(item_ids), x_user.shape[1]), dtype=torch.float)

    # Concatenate user and item features
    x = torch.cat([x_user, x_item], dim=0)
    return x

def create_pyg_data_object(x, edge_index):
    """
    Create PyG Data object from features and edge_index.
    
    Args:
        x (torch.FloatTensor): Node feature matrix.
        edge_index (torch.LongTensor): Edge indices.
        
    Returns:
        torch_geometric.data.Data: PyG data object ready for training.
    """
    data = Data(x=x, edge_index=edge_index)
    return data