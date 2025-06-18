import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from matplotlib import gridspec
from collections import Counter
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

def prepare_data_for_pca(customer_data_scaled):
    """
    Prepare data for PCA by setting CustomerID as index.
    
    Parameters:
    customer_data_scaled (DataFrame): Scaled customer data with CustomerID column
    
    Returns:
    DataFrame: Prepared data with CustomerID as index
    """
    return customer_data_scaled.set_index('CustomerID')

def perform_pca_analysis(data, optimal_k=6):
    """
    Perform PCA analysis and calculate explained variance metrics.
    
    Parameters:
    data (DataFrame): Prepared data for PCA
    optimal_k (int): Optimal number of components to highlight
    
    Returns:
    tuple: (PCA object, explained_variance_ratio, cumulative_explained_variance)
    """
    pca = PCA().fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    return pca, explained_variance_ratio, cumulative_explained_variance

def plot_variance_explained(explained_variance_ratio, cumulative_explained_variance, optimal_k=6):
    """
    Plot the variance explained by PCA components.
    
    Parameters:
    explained_variance_ratio (array): Explained variance ratio for each component
    cumulative_explained_variance (array): Cumulative explained variance
    optimal_k (int): Optimal number of components to highlight
    """
    # Set seaborn plot style
    sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

    # Create figure
    plt.figure(figsize=(20, 10))

    # Bar chart for the explained variance of each component
    barplot = sns.barplot(x=list(range(1, len(cumulative_explained_variance) + 1)),
                        y=explained_variance_ratio,
                        color='#fcc36d',
                        alpha=0.8)

    # Line plot for the cumulative explained variance
    lineplot, = plt.plot(range(0, len(cumulative_explained_variance)), cumulative_explained_variance,
                        marker='o', linestyle='--', color='#ff6200', linewidth=2)

    # Plot optimal k value line
    optimal_k_line = plt.axvline(optimal_k - 1, color='red', linestyle='--', 
                               label=f'Optimal k value = {optimal_k}') 

    # Set labels and title
    plt.xlabel('Number of Components', fontsize=14)
    plt.ylabel('Explained Variance', fontsize=14)
    plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

    # Customize ticks and legend
    plt.xticks(range(0, len(cumulative_explained_variance)))
    plt.legend(handles=[barplot.patches[0], lineplot, optimal_k_line],
            labels=['Explained Variance of Each Component', 
                   'Cumulative Explained Variance', 
                   f'Optimal k value = {optimal_k}'],
            loc=(0.62, 0.1),
            frameon=True,
            framealpha=1.0,  
            edgecolor='#ff6200')  

    # Display variance values on plot
    x_offset = -0.3
    y_offset = 0.01
    for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, 
                                                   cumulative_explained_variance)):
        plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
        if i > 0:
            plt.text(i + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", 
                    ha="center", va="bottom", fontsize=10)

    plt.grid(axis='both')   
    plt.show()

def apply_pca_transformation(data, n_components=6):
    """
    Apply PCA transformation with specified number of components.
    
    Parameters:
    data (DataFrame): Data to transform
    n_components (int): Number of principal components to keep
    
    Returns:
    DataFrame: Transformed data with principal components
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca_data, 
                         columns=[f'PC{i+1}' for i in range(n_components)],
                         index=data.index)
    return pca_df

def dimension_reduction_pca(customer_data_scaled, optimal_k=6):
    """
    Complete PCA workflow with visualization and transformation.
    
    Parameters:
    customer_data_scaled (DataFrame): Scaled customer data with CustomerID column
    optimal_k (int): Optimal number of components to use
    
    Returns:
    DataFrame: Data transformed with PCA
    """
    # Prepare data
    prepared_data = prepare_data_for_pca(customer_data_scaled)
    
    # Perform PCA analysis
    pca, explained_var, cum_explained_var = perform_pca_analysis(prepared_data, optimal_k)
    
    # Plot results
    plot_variance_explained(explained_var, cum_explained_var, optimal_k)
    
    # Apply PCA transformation
    pca_data = apply_pca_transformation(prepared_data, optimal_k)
    
    return pca_data

def calculate_silhouette_scores(df, start_k, stop_k):
    """
    Calculate silhouette scores for a range of k values.
    
    Parameters:
    df (DataFrame): Data to cluster
    start_k (int): Minimum number of clusters to evaluate
    stop_k (int): Maximum number of clusters to evaluate
    
    Returns:
    tuple: (silhouette_scores, best_k)
    """
    silhouette_scores = []
    
    for k in range(start_k, stop_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=0)
        labels = km.fit_predict(df)
        score = silhouette_score(df, labels)
        silhouette_scores.append(score)
    
    best_k = start_k + np.argmax(silhouette_scores)
    return silhouette_scores, best_k

def plot_silhouette_scores(silhouette_scores, start_k, stop_k, best_k):
    """
    Plot the average silhouette scores across different k values.
    
    Parameters:
    silhouette_scores (list): Silhouette scores for each k
    start_k (int): Minimum number of clusters evaluated
    stop_k (int): Maximum number of clusters evaluated
    best_k (int): Optimal number of clusters
    """
    sns.set_palette(['darkorange'])
    
    plt.plot(range(start_k, stop_k + 1), silhouette_scores, marker='o')
    plt.xticks(range(start_k, stop_k + 1))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette score')
    plt.title('Average Silhouette Score for Different k Values', fontsize=15)
    
    optimal_k_text = f'The k value with the highest Silhouette score is: {best_k}'
    plt.text(10, 0.23, optimal_k_text, fontsize=12, 
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(facecolor='#fcc36d', edgecolor='#ff6200', 
                      boxstyle='round, pad=0.5'))

def plot_individual_silhouettes(df, start_k, stop_k, grid):
    """
    Plot silhouette visualizations for each k value.
    
    Parameters:
    df (DataFrame): Data to cluster
    start_k (int): Minimum number of clusters to visualize
    stop_k (int): Maximum number of clusters to visualize
    grid (GridSpec): Grid layout for subplots
    """
    colors = sns.color_palette("bright")
    
    for i in range(start_k, stop_k + 1):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
        row_idx, col_idx = divmod(i - start_k, 2)
        
        ax = plt.subplot(grid[row_idx + 1, col_idx])
        visualizer = SilhouetteVisualizer(km, colors=colors, ax=ax)
        visualizer.fit(df)
        
        score = silhouette_score(df, km.labels_)
        ax.text(0.97, 0.02, f'Silhouette Score: {score:.2f}', 
                fontsize=12, ha='right', transform=ax.transAxes, color='red')
        ax.set_title(f'Silhouette Plot for {i} Clusters', fontsize=15)

def silhouette_analysis(df, start_k, stop_k, figsize=(15, 16)):
    """
    Perform complete silhouette analysis workflow with visualization.
    
    Parameters:
    df (DataFrame): Data to analyze
    start_k (int): Minimum number of clusters to evaluate
    stop_k (int): Maximum number of clusters to evaluate
    figsize (tuple): Figure dimensions
    """
    # Set up figure and grid
    plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(stop_k - start_k + 1, 2)
    
    # First plot (average scores)
    plt.subplot(grid[0, :])
    scores, best_k = calculate_silhouette_scores(df, start_k, stop_k)
    plot_silhouette_scores(scores, start_k, stop_k, best_k)
    
    # Second plots (individual silhouettes)
    plot_individual_silhouettes(df, start_k, stop_k, grid)
    
    plt.tight_layout()
    plt.show()

from collections import Counter
import time
from sklearn.cluster import KMeans

def perform_kmeans_clustering(data, n_clusters=3, random_state=0):
    """
    Perform KMeans clustering and measure training time.
    
    Parameters:
    data (DataFrame/np.array): Data to cluster
    n_clusters (int): Number of clusters
    random_state (int): Random seed
    
    Returns:
    tuple: (kmeans model, labels, training_time)
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                   n_init=10, max_iter=100, random_state=random_state)
    
    start_time = time.time()
    kmeans.fit(data)
    training_time = time.time() - start_time
    
    return kmeans, kmeans.labels_, training_time

def remap_cluster_labels(labels, mapping=None):
    """
    Remap cluster labels according to specified mapping or frequency.
    
    Parameters:
    labels (array): Original cluster labels
    mapping (dict): Optional custom mapping (if None, sorts by frequency)
    
    Returns:
    array: Remapped labels
    """
    if mapping is None:
        # Create mapping based on frequency (most common gets label 0)
        cluster_freq = Counter(labels)
        mapping = {label: new_label for new_label, (label, _) 
                  in enumerate(cluster_freq.most_common())}
    
    return np.array([mapping[label] for label in labels])

def add_clusters_to_data(original_data, pca_data, labels, cluster_col='cluster'):
    """
    Add cluster labels to both original and PCA datasets.
    
    Parameters:
    original_data (DataFrame): Original dataset
    pca_data (DataFrame): PCA-transformed dataset
    labels (array): Cluster labels
    cluster_col (str): Name for cluster column
    
    Returns:
    tuple: (original_data_with_clusters, pca_data_with_clusters)
    """
    original_data = original_data.copy()
    pca_data = pca_data.copy()
    
    original_data[cluster_col] = labels
    pca_data[cluster_col] = labels
    
    return original_data, pca_data

def clustering(original_data, pca_data, n_clusters=3, custom_mapping=None):
    """
    Complete clustering workflow with optional label remapping.
    
    Parameters:
    original_data (DataFrame): Original dataset
    pca_data (DataFrame): PCA-transformed dataset
    n_clusters (int): Number of clusters
    custom_mapping (dict): Optional custom label mapping
    
    Returns:
    tuple: (kmeans_model, original_data_with_clusters, 
           pca_data_with_clusters, training_time)
    """
    # Perform clustering
    kmeans, labels, training_time = perform_kmeans_clustering(
        pca_data, n_clusters=n_clusters)
    
    # Remap labels (using custom mapping if provided)
    if custom_mapping is None:
        # Default mapping example (adjust as needed)
        custom_mapping = {0: 2, 1: 0, 2: 1}
    
    remapped_labels = remap_cluster_labels(labels, custom_mapping)
    
    # Add clusters to datasets
    original_data, pca_data = add_clusters_to_data(
        original_data, pca_data, remapped_labels)
    
    return kmeans, original_data, pca_data, training_time

def calculate_cluster_percentages(cluster_series):
    """
    Calculate percentage distribution of clusters.
    
    Parameters:
    cluster_series (Series): Cluster labels
    
    Returns:
    DataFrame: Cluster percentages with columns ['Cluster', 'Percentage']
    """
    percentages = (cluster_series.value_counts(normalize=True) * 100).reset_index()
    percentages.columns = ['Cluster', 'Percentage']
    return percentages.sort_values(by='Cluster')

def plot_cluster_distribution(cluster_percentages, colors=None, figsize=(10, 4)):
    """
    Visualize cluster distribution as horizontal bar plot.
    
    Parameters:
    cluster_percentages (DataFrame): Cluster percentages from calculate_cluster_percentages
    colors (list): Color palette for clusters
    figsize (tuple): Figure dimensions
    """
    if colors is None:
        colors = ['#e8000b', '#1ac938', '#023eff']  # Default color scheme
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Percentage', y='Cluster', 
                    data=cluster_percentages, 
                    orient='h', 
                    palette=colors)
    
    # Add percentage labels
    for index, value in enumerate(cluster_percentages['Percentage']):
        ax.text(value + 0.5, index, f'{value:.2f}%', va='center')
    
    ax.set_title('Distribution of Customers Across Clusters', fontsize=14)
    ax.set_xlabel('Percentage (%)')
    ax.set_xticks(np.arange(0, 50, 5))
    plt.show()

def visualize_clusters(cluster_series, colors=None, figsize=(10, 4)):
    """
    Complete workflow for cluster distribution visualization.
    
    Parameters:
    cluster_series (Series): Cluster labels
    colors (list): Optional custom color palette
    figsize (tuple): Figure dimensions
    """
    percentages = calculate_cluster_percentages(cluster_series)
    plot_cluster_distribution(percentages, colors, figsize)

def evaluate_clustering(customer_data_pca):
    """
    Evaluates the clustering quality using common metrics:
    Silhouette Score, Calinski-Harabasz Score, and Davies-Bouldin Score.

    Parameters:
        customer_data_pca (pd.DataFrame): DataFrame containing PCA-reduced features
                                          and a 'cluster' column indicating cluster labels.

    Returns:
        None: Prints a table of clustering evaluation metrics.
    """
    # Compute number of customers
    num_observations = len(customer_data_pca)

    # Separate features and cluster labels
    X = customer_data_pca.drop('cluster', axis=1)
    clusters = customer_data_pca['cluster']

    # Compute clustering metrics
    sil_score = silhouette_score(X, clusters)
    calinski_score = calinski_harabasz_score(X, clusters)
    davies_score = davies_bouldin_score(X, clusters)

    # Display metrics in a formatted table
    table_data = [
        ["Number of Observations", num_observations],
        ["Silhouette Score", sil_score],
        ["Calinski-Harabasz Score", calinski_score],
        ["Davies-Bouldin Score", davies_score]
    ]

    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt='pretty'))

#refactored radar chart visualization
def prepare_cluster_data(df, cluster_col='cluster', index_col='CustomerID'):
    """
    Prepare data for cluster visualization by standardizing features and keeping clusters.
    
    Parameters:
    df (DataFrame): Original data with cluster assignments
    cluster_col (str): Name of cluster column
    index_col (str): Name of index column
    
    Returns:
    DataFrame: Standardized data with cluster assignments
    """
    # Set index and standardize features
    df_indexed = df.set_index(index_col)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_indexed.drop(columns=[cluster_col]))
    
    # Create new DataFrame with standardized values
    df_standardized = pd.DataFrame(
        standardized_data,
        columns=df_indexed.columns[:-1],  # Exclude cluster column
        index=df_indexed.index
    )
    df_standardized[cluster_col] = df_indexed[cluster_col]
    
    return df_standardized

def calculate_cluster_centroids(df_standardized, cluster_col='cluster'):
    """
    Calculate mean feature values for each cluster.
    
    Parameters:
    df_standardized (DataFrame): Standardized data with cluster assignments
    cluster_col (str): Name of cluster column
    
    Returns:
    DataFrame: Cluster centroids
    """
    return df_standardized.groupby(cluster_col).mean()

def create_radar_chart(ax, angles, data, color, cluster, title_size=20):
    """
    Create a single radar chart for a cluster.
    
    Parameters:
    ax (matplotlib axis): Axis to plot on
    angles (list): Angles for each axis
    data (list): Data values to plot
    color (str): Color for the plot
    cluster (int): Cluster number
    title_size (int): Font size for title
    """
    # Plot the data and fill the area
    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
    
    # Add title
    ax.set_title(f'Cluster {cluster}', size=title_size, color=color, y=1.1)

def setup_radar_axes(ax, angles, labels):
    """
    Configure radar chart axes and labels.
    
    Parameters:
    ax (matplotlib axis): Axis to configure
    angles (list): Angles for each axis
    labels (list): Labels for each axis
    """
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    ax.grid(color='grey', linewidth=0.5)

def visualize_cluster_profiles(df, colors=None, figsize=(20, 10)):
    """
    Create radar charts to visualize cluster profiles.
    
    Parameters:
    df (DataFrame): Original data with cluster assignments
    colors (list): Color palette for clusters
    figsize (tuple): Figure dimensions
    """
    # Set default colors
    if colors is None:
        colors = ['#e8000b', '#1ac938', '#023eff']  # Red, Green, Blue
    
    # Prepare data
    df_standardized = prepare_cluster_data(df)
    centroids = calculate_cluster_centroids(df_standardized)
    
    # Set up radar chart parameters
    labels = np.array(centroids.columns)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop for circular plot
    labels = np.concatenate((labels, [labels[0]]))
    angles += angles[:1]
    
    # Initialize figure
    fig, axes = plt.subplots(
        figsize=figsize,
        subplot_kw=dict(polar=True),
        nrows=1,
        ncols=len(centroids)
    )
    
    # Create radar chart for each cluster
    for i, (color, ax) in enumerate(zip(colors, axes)):
        data = centroids.loc[i].tolist()
        data += data[:1]  # Complete the loop
        create_radar_chart(ax, angles, data, color, i)
        setup_radar_axes(ax, angles, labels)
    
    plt.tight_layout()
    plt.show()

# The following functions are related to the recommendation system
def filter_outliers(main_df, outliers_df, id_col='CustomerID'):
    """
    Remove outlier customers from the main dataframe.
    
    Parameters:
    main_df (DataFrame): Main transaction data
    outliers_df (DataFrame): Data containing outlier customer IDs
    id_col (str): Name of customer ID column
    
    Returns:
    DataFrame: Filtered dataframe without outliers
    """
    outlier_ids = outliers_df[id_col].astype('float').unique()
    return main_df[~main_df[id_col].isin(outlier_ids)]

def merge_cluster_info(transactions_df, customer_df, id_col='CustomerID', cluster_col='cluster'):
    """
    Merge transaction data with customer cluster information.
    
    Parameters:
    transactions_df (DataFrame): Filtered transaction data
    customer_df (DataFrame): Customer data with cluster assignments
    id_col (str): Name of customer ID column
    cluster_col (str): Name of cluster column
    
    Returns:
    DataFrame: Merged data with cluster information
    """
    # Ensure consistent data types
    customer_df[id_col] = customer_df[id_col].astype('float')
    
    return transactions_df.merge(
        customer_df[[id_col, cluster_col]],
        on=id_col,
        how='inner'
    )

def get_top_products(merged_data, n_top=10, cluster_col='cluster', 
                    product_cols=['StockCode', 'Description'], quantity_col='Quantity'):
    """
    Identify top selling products per cluster.
    
    Parameters:
    merged_data (DataFrame): Data with transactions and cluster info
    n_top (int): Number of top products to return per cluster
    cluster_col (str): Name of cluster column
    product_cols (list): Columns identifying products
    quantity_col (str): Name of quantity column
    
    Returns:
    DataFrame: Top products per cluster
    """
    # Group by cluster and product, sum quantities
    product_sales = merged_data.groupby([cluster_col] + product_cols)[quantity_col].sum().reset_index()
    
    # Sort and get top products
    product_sales = product_sales.sort_values(by=[cluster_col, quantity_col], ascending=[True, False])
    return product_sales.groupby(cluster_col).head(n_top)

def get_customer_purchases(merged_data, id_col='CustomerID', cluster_col='cluster',
                         product_col='StockCode', quantity_col='Quantity'):
    """
    Create record of products purchased by each customer.
    
    Parameters:
    merged_data (DataFrame): Data with transactions and cluster info
    id_col (str): Customer ID column
    cluster_col (str): Cluster column
    product_col (str): Product ID column
    quantity_col (str): Quantity column
    
    Returns:
    DataFrame: Customer purchase history
    """
    return merged_data.groupby([id_col, cluster_col, product_col])[quantity_col].sum().reset_index()

def generate_recommendations(top_products, customer_purchases, customer_data, 
                           n_rec=3, cluster_col='cluster', id_col='CustomerID',
                           product_cols=['StockCode', 'Description']):
    """
    Generate product recommendations for each customer.
    
    Parameters:
    top_products (DataFrame): Top products per cluster
    customer_purchases (DataFrame): Customer purchase history
    customer_data (DataFrame): Original customer data
    n_rec (int): Number of recommendations per customer
    cluster_col (str): Cluster column name
    id_col (str): Customer ID column name
    product_cols (list): Product identifier columns
    
    Returns:
    DataFrame: Recommendations for each customer
    """
    recommendations = []
    
    for cluster in top_products[cluster_col].unique():
        cluster_top_products = top_products[top_products[cluster_col] == cluster]
        cluster_customers = customer_data[customer_data[cluster_col] == cluster][id_col]
        
        for customer in cluster_customers:
            # Get products already purchased by customer
            purchased = customer_purchases[
                (customer_purchases[id_col] == customer) & 
                (customer_purchases[cluster_col] == cluster)
            ][product_cols[0]].tolist()
            
            # Find top products not purchased
            not_purchased = cluster_top_products[~cluster_top_products[product_cols[0]].isin(purchased)]
            top_recs = not_purchased.head(n_rec)
            
            # Prepare recommendation record
            rec_record = [customer, cluster]
            for _, row in top_recs.iterrows():
                rec_record.extend(row[product_cols].tolist())
            
            # Pad with None if fewer than n_rec recommendations
            rec_record += [None] * (n_rec * len(product_cols) - len(rec_record) + 2)
            recommendations.append(rec_record)
    
    # Create column names for recommendations
    rec_cols = [id_col, cluster_col]
    for i in range(1, n_rec+1):
        for col in product_cols:
            rec_cols.append(f'Rec{i}_{col}')
    
    return pd.DataFrame(recommendations, columns=rec_cols)

def get_recommendation_system(df_diy, outliers_data, customer_data_cleaned):
    """
    Complete recommendation system workflow.
    
    Parameters:
    df_diy (DataFrame): Original transaction data
    outliers_data (DataFrame): Outlier customer data
    customer_data_cleaned (DataFrame): Customer data with clusters
    
    Returns:
    DataFrame: Customer data with recommendations
    """
    # Step 1: Filter outliers
    df_filtered = filter_outliers(df_diy, outliers_data)
    
    # Step 2-3: Merge with cluster info
    merged_data = merge_cluster_info(df_filtered, customer_data_cleaned)
    
    # Step 4: Get top products
    top_products = get_top_products(merged_data)
    
    # Step 5: Get customer purchases
    customer_purchases = get_customer_purchases(merged_data)
    
    # Step 6: Generate recommendations
    recommendations_df = generate_recommendations(top_products, customer_purchases, customer_data_cleaned)
    
    # Step 7: Merge with original data
    return customer_data_cleaned.merge(recommendations_df, on=['CustomerID', 'cluster'], how='right')