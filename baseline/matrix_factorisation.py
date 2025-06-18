from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import time

def train_svd_recommender(df, n_components=20):
    """
    Train an SVD model using binary customer-item interactions.

    Parameters:
    - df (DataFrame): Original transaction data containing 'CustomerID', 'StockCode', and 'Quantity'.
    - n_components (int): Number of latent factors.

    Returns:
    - score_df (DataFrame): Predicted customer-item preference scores.
    - training_time (float): Time taken to train and compute predictions.
    - customer_factors (DataFrame): Customer latent vectors.
    - item_factors (DataFrame): Item latent vectors.
    """
    # Copy dataset
    df_copy = df.copy()

    # Create binary interaction matrix (1 if item was purchased at least once)
    interaction_df = df_copy.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().reset_index()
    interaction_df['interaction'] = 1  # could replace with 'Quantity' or 'TotalPrice' for weighted scoring

    # Pivot to get user-item matrix
    user_item_matrix = interaction_df.pivot(index='CustomerID', columns='StockCode', values='interaction').fillna(0)

    # Ensure column names are strings (for compatibility)
    user_item_matrix.columns = user_item_matrix.columns.astype(str)

    # Initialize SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # Start timing
    start_time = time.time()

    # Fit SVD model and get customer/item latent representations
    latent_matrix = svd.fit_transform(user_item_matrix)
    customer_factors = pd.DataFrame(latent_matrix, index=user_item_matrix.index)
    item_factors = pd.DataFrame(svd.components_.T, index=user_item_matrix.columns)

    # Compute score matrix (dot product)
    score_matrix = np.dot(customer_factors.values, item_factors.values.T)
    score_df = pd.DataFrame(score_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

    # End timing
    training_time = time.time() - start_time

    return score_df, training_time, customer_factors, item_factors

def get_top_k_recommendations(customer_id, score_df, k=3):
    """
    Retrieve top-k recommended items for a specific customer.

    Parameters:
    - customer_id: ID of the customer to generate recommendations for.
    - score_df (DataFrame): Matrix of predicted scores (index = CustomerID, columns = StockCode).
    - k (int): Number of items to recommend.

    Returns:
    - List of StockCode strings representing top-k recommended items.
    """
    try:
        # Get the customer's predicted scores for all items
        customer_scores = score_df.loc[customer_id]

        # Sort the scores descending and return top-k item codes
        top_k_items = customer_scores.sort_values(ascending=False).head(k).index.tolist()
        return top_k_items

    except KeyError:
        # If customer ID is not in the score_df, return an empty list
        return []


def generate_recommendations(future_baskets, score_df, k=3):
    """
    Generate top-k recommendations for each basket in future_baskets.

    Args:
        future_baskets (DataFrame): Contains columns ['BasketID', 'CustomerID', 'StockCode'].
        score_df (DataFrame): Predicted score matrix (customers x items).
        k (int): Number of recommendations to generate.

    Returns:
        DataFrame: Each row contains BasketID, CustomerID, and list of RecommendedItems.
    """
    # Group to get list of actual purchased items per basket
    basket_groups = future_baskets.groupby(['BasketID', 'CustomerID'])['StockCode'].apply(set).reset_index()

    results = []
    for _, row in basket_groups.iterrows():
        basket_id = row['BasketID']
        customer_id = row['CustomerID']
        actual_items = row['StockCode']

        # Get all recommendations for the customer
        recommended_items = get_top_k_recommendations(customer_id, score_df, k=20)  # get more for filtering

        # Filter out items already in the basket
        filtered_recommendations = [item for item in recommended_items if item not in actual_items]

        # Keep only top-k after filtering
        final_recommendations = filtered_recommendations[:k]

        results.append({
            'BasketID': basket_id,
            'CustomerID': customer_id,
            'RecommendedItems': final_recommendations,
        })

    return pd.DataFrame(results)

