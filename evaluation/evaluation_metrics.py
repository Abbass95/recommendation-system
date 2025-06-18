# Core Data Handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from statsmodels.stats.proportion import proportions_ztest


def compare_customer_sets(past_customers_df, future_customers_df, id_col='CustomerID'):
    """
    Compare customer sets between past and future data.
    
    Parameters:
    past_customers_df (DataFrame): DataFrame with past customer data
    future_customers_df (DataFrame): DataFrame with future customer data
    id_col (str): Name of customer ID column
    
    Returns:
    tuple: (returning_customers_set, new_customers_set, future_customers_set)
    """
    # Ensure consistent data types
    past_customers_df[id_col] = past_customers_df[id_col].astype(float)
    future_customers_df[id_col] = future_customers_df[id_col].astype(float)
    
    # Extract unique customer IDs
    past_customers = set(past_customers_df[id_col].dropna().unique())
    future_customers = set(future_customers_df[id_col].dropna().unique())
    
    # Compare sets
    returning_customers = future_customers.intersection(past_customers)
    new_customers = future_customers.difference(past_customers)
    
    return returning_customers, new_customers, future_customers

def print_customer_stats(returning_customers, new_customers, future_customers):
    """
    Print statistics about customer cohorts.
    
    Parameters:
    returning_customers (set): Set of returning customer IDs
    new_customers (set): Set of new customer IDs
    future_customers (set): Set of all future customer IDs
    """
    print("🔁 Number of returning customers:", len(returning_customers))
    print("🆕 Number of new customers:", len(new_customers))
    print("📊 Total future customers:", len(future_customers))
    print("🔄 Retention rate: {:.1f}%".format(
        len(returning_customers) / len(future_customers) * 100 if future_customers else 0
    ))

def check_new_customers(past_customers_df, future_customers_df, id_col='CustomerID'):
    """
    Analyze customer cohorts between past and future data.
    
    Parameters:
    past_customers_df (DataFrame): DataFrame with past customer data
    future_customers_df (DataFrame): DataFrame with future customer data
    id_col (str): Name of customer ID column
    
    Returns:
    tuple: (returning_customers_df, new_customers_df)
    """
    # Compare customer sets
    returning_customers, new_customers, future_customers = compare_customer_sets(
        past_customers_df, future_customers_df, id_col
    )
    
    # Print statistics
    print_customer_stats(returning_customers, new_customers, future_customers)
    
    # Return filtered DataFrames
    returning_df = future_customers_df[future_customers_df[id_col].isin(returning_customers)]
    new_df = future_customers_df[future_customers_df[id_col].isin(new_customers)]
    
    return returning_df, new_df

def recall_at_3(future_baskets, recommendations_flat):
    """
    Computes Recall@3 for each basket by comparing actual purchased items with top-3 recommended items.

    Parameters:
        future_baskets (pd.DataFrame): Contains 'BasketID', 'CustomerID', and 'StockCode' columns
                                       representing the actual future purchases.
        recommendations_flat (pd.DataFrame): Contains 'CustomerID' and 'StockCode' columns with
                                             recommended items per customer.

    Returns:
        pd.DataFrame: Evaluation DataFrame with Recall@3 per (BasketID, CustomerID) pair.
    """
    # Step 1: Get actual items per basket (ground truth)
    ground_truth = future_baskets.groupby(['BasketID', 'CustomerID'])['StockCode'].apply(set).reset_index(name='true_items')

    # Step 2: Get predicted items per customer
    recommended_items = recommendations_flat.groupby('CustomerID')['StockCode'].apply(list).reset_index(name='predicted_items')

    # Step 3: Join ground truth and predictions
    evaluation_df = ground_truth.merge(recommended_items, on='CustomerID', how='left')

    # Step 4: Compute Recall@3 for each row
    def recall_at_k(true_items, predicted_items, k=3):
        if not isinstance(predicted_items, list) or not predicted_items:
            return 0.0
        hits = len(set(predicted_items[:k]) & true_items)
        return hits / len(true_items) if true_items else 0.0

    evaluation_df['Recall@3'] = evaluation_df.apply(
        lambda row: recall_at_k(row['true_items'], row['predicted_items'], k=3),
        axis=1
    )

    return evaluation_df

def ndcg_at_k(true_items, predicted_items, k=3):
    """
    Computes Normalized Discounted Cumulative Gain at rank k (NDCG@k).

    Parameters:
        true_items (set): Set of actual relevant items.
        predicted_items (list): List of predicted items in ranked order.
        k (int): Rank cutoff.

    Returns:
        float: NDCG score.
    """
    if not isinstance(predicted_items, list) or not predicted_items:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(predicted_items[:k]):
        if item in true_items:
            dcg += 1 / np.log2(i + 2)  # i+2 because log2(1) = 0

    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def average_precision_at_k(true_items, predicted_items, k=3):
    """
    Computes Average Precision at rank k (AP@k).

    Parameters:
        true_items (set): Set of actual relevant items.
        predicted_items (list): Ranked list of predicted items.
        k (int): Rank cutoff.

    Returns:
        float: Average Precision at k.
    """
    if not isinstance(predicted_items, list) or not predicted_items:
        return 0.0

    score = 0.0
    hits = 0
    for i, item in enumerate(predicted_items[:k]):
        if item in true_items:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(true_items), k) if true_items else 0.0

def random_hit_prob(future_baskets):
    """
    Computes the probability of randomly hitting a relevant item
    in a basket (Random Hit Probability).

    Parameters:
        future_baskets (DataFrame): DataFrame with at least 'BasketID',
                                    'CustomerID', and 'StockCode' columns.

    Returns:
        float: Random hit probability.
    """
    # Ensure required columns exist
    required_cols = {'BasketID', 'CustomerID', 'StockCode'}
    if not required_cols.issubset(future_baskets.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Total number of unique products
    total_unique_products = future_baskets['StockCode'].nunique()

    # Average number of items per basket
    avg_items_per_basket = future_baskets.groupby(['BasketID', 'CustomerID'])['StockCode'].nunique().mean()

    # Avoid division by zero
    if total_unique_products == 0:
        return 0.0

    # Random hit probability
    return avg_items_per_basket / total_unique_products

def hits_at_k(true_items: set, predicted_items: list, k: int = 3) -> int:
    """
    Computes the number of correct items (hits) in the top-k predicted items.

    Parameters:
        true_items (set): Set of ground truth items.
        predicted_items (list): List of predicted items (ranked).
        k (int): Number of top predicted items to consider.

    Returns:
        int: Number of hits in the top-k predicted items.
    """
    if not predicted_items or not true_items:
        return 0
    return len(set(predicted_items[:k]) & true_items)

def compute_lift_p_value(evaluation_df, future_baskets, k=3):
    """
    Compute p-value for Lift@K using a one-sided proportions z-test.

    Parameters:
    - evaluation_df (DataFrame): Must contain column 'Hits@K' (e.g., 'Hits@3').
    - future_baskets (DataFrame): Ground truth basket data to compute random hit probability.
    - k (int): Top-K cutoff used for evaluation.

    Returns:
    - p-value (float): Result of one-sided z-test (model hit rate > random hit rate).
    """
    # Compute baseline hit probability
    baseline_rate = random_hit_prob(future_baskets)

    # Count how many baskets had at least one hit in top-K
    hit_col = f'Hits@{k}'
    if hit_col not in evaluation_df.columns:
        raise ValueError(f"Column '{hit_col}' not found in evaluation DataFrame.")
    
    hit_count = (evaluation_df[hit_col] > 0).sum()
    total_baskets = len(evaluation_df)

    # Perform one-sided proportions z-test
    stat, p_value = proportions_ztest(
        count=hit_count,
        nobs=total_baskets,
        value=baseline_rate,
        alternative='larger'  # One-sided: testing model > random
    )

    return p_value

