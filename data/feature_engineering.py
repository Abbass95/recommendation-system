import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# RFM features
def days_last_purchase(df_diy: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of days since the last purchase for each customer.

    Parameters:
        df_diy (pd.DataFrame): DataFrame containing 'CustomerID' and 'InvoiceDate' columns.

    Returns:
        pd.DataFrame: DataFrame with 'CustomerID' and 'Days_Since_Last_Purchase'.
    """
    if 'CustomerID' not in df_diy.columns or 'InvoiceDate' not in df_diy.columns:
        raise ValueError("The input DataFrame must contain 'CustomerID' and 'InvoiceDate' columns.")
    
    # Ensure datetime format
    df_diy['InvoiceDate'] = pd.to_datetime(df_diy['InvoiceDate'])

    # Extract only the date part for daily grouping
    df_diy['InvoiceDay'] = df_diy['InvoiceDate'].dt.date

    # Most recent purchase per customer
    customer_last_purchase = df_diy.groupby('CustomerID')['InvoiceDay'].max().reset_index()

    # Calculate days since most recent date in dataset
    most_recent_date = pd.to_datetime(df_diy['InvoiceDay'].max())
    customer_last_purchase['InvoiceDay'] = pd.to_datetime(customer_last_purchase['InvoiceDay'])
    customer_last_purchase['Days_Since_Last_Purchase'] = (
        most_recent_date - customer_last_purchase['InvoiceDay']
    ).dt.days

    return customer_last_purchase[['CustomerID', 'Days_Since_Last_Purchase']]

def frequency_features(df: pd.DataFrame, customer_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds frequency-related features to customer data:
    - Total number of unique baskets
    - Total number of products purchased

    Parameters:
        df (pd.DataFrame): Transaction-level dataset with 'CustomerID', 'BasketID', and 'Quantity'.
        customer_data (pd.DataFrame): Existing customer-level dataframe to be enriched.

    Returns:
        tuple:
            - pd.DataFrame: Updated customer_data with frequency features.
            - pd.DataFrame: total_baskets (CustomerID + Total_Baskets)
    """
    # Validate required columns
    required_cols = ['CustomerID', 'BasketID', 'Quantity']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"The dataframe must contain columns: {required_cols}")

    # Total unique baskets per customer
    total_baskets = df.groupby('CustomerID')['BasketID'].nunique().reset_index()
    total_baskets.rename(columns={'BasketID': 'Total_Baskets'}, inplace=True)

    # Total quantity of products purchased per customer
    total_products_purchased = df.groupby('CustomerID')['Quantity'].sum().reset_index()
    total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

    # Merge features into customer_data
    customer_data = customer_data.merge(total_baskets, on='CustomerID', how='left')
    customer_data = customer_data.merge(total_products_purchased, on='CustomerID', how='left')

    return customer_data, total_baskets


def monetary_features(
    df_diy: pd.DataFrame, 
    customer_data: pd.DataFrame, 
    total_baskets: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes and adds monetary-related features for each customer:
    - Total spend
    - Average basket value

    Parameters:
        df_diy (pd.DataFrame): Transaction data with 'UnitPrice', 'Quantity', and 'CustomerID'.
        customer_data (pd.DataFrame): Customer-level data to be updated.
        total_baskets (pd.DataFrame): DataFrame with 'CustomerID' and 'Total_baskets'.

    Returns:
        pd.DataFrame: Updated customer_data with monetary features.
    """
    required_cols = ['CustomerID', 'UnitPrice', 'Quantity']
    if not all(col in df_diy.columns for col in required_cols):
        raise ValueError(f"df_diy must contain: {required_cols}")

    # Calculate total spend
    df_diy = df_diy.copy()
    df_diy['Total_Spend'] = df_diy['UnitPrice'] * df_diy['Quantity']
    total_spend = df_diy.groupby('CustomerID')['Total_Spend'].sum().reset_index()

    # Merge to get average basket value
    avg_basket = total_spend.merge(total_baskets, on='CustomerID')
    avg_basket['Average_Basket_Value'] = avg_basket['Total_Spend'] / avg_basket['Total_baskets'].replace(0, np.nan)

    # Merge new features into customer_data
    customer_data = customer_data.merge(total_spend, on='CustomerID', how='left')
    customer_data = customer_data.merge(avg_basket[['CustomerID', 'Average_Basket_Value']], on='CustomerID', how='left')

    return customer_data

def unique_product(
    df: pd.DataFrame,
    customer_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds the number of unique products purchased by each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction‑level dataset containing 'CustomerID' and 'StockCode'.
    customer_data : pd.DataFrame
        Customer‑level DataFrame to enrich.

    Returns
    -------
    tuple
        (updated_customer_data, unique_products_purchased)
    """
    # --- validation ---------------------------------------------------------
    required_cols = ['CustomerID', 'StockCode']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df must contain columns {required_cols}")

    # --- feature engineering ------------------------------------------------
    unique_products_purchased = (
        df.groupby('CustomerID')['StockCode']
          .nunique()
          .reset_index()
          .rename(columns={'StockCode': 'Unique_Products_Purchased'})
    )

    # --- merge into customer_data ------------------------------------------
    updated_customer_data = customer_data.merge(
        unique_products_purchased, on='CustomerID', how='left'
    )

    return updated_customer_data, unique_products_purchased

# Behavioral features
def behavioral_features(df: pd.DataFrame, customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts behavioral features for each customer and merges them into customer_data.

    Parameters:
        df (pd.DataFrame): Transaction-level dataset with 'CustomerID' and 'InvoiceDate'.
        customer_data (pd.DataFrame): Customer-level dataframe to enrich.

    Returns:
        pd.DataFrame: Updated customer_data with new behavioral features.
    """
    required_cols = ['CustomerID', 'InvoiceDate']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"The dataframe must contain columns: {required_cols}")
    
    # Work on a copy to avoid changing original df
    df_copy = df.copy()
    df_copy['InvoiceDate'] = pd.to_datetime(df_copy['InvoiceDate'])
    
    # Extract day of week and hour
    df_copy['Day_Of_Week'] = df_copy['InvoiceDate'].dt.dayofweek
    df_copy['Hour'] = df_copy['InvoiceDate'].dt.hour
    
    # Extract date only for diff calculations
    df_copy['InvoiceDay'] = df_copy['InvoiceDate'].dt.date
    
    # Average days between consecutive purchases per customer
    days_between = (
        df_copy.groupby('CustomerID')['InvoiceDay']
        .apply(lambda x: x.diff().dropna().dt.days if pd.api.types.is_datetime64_any_dtype(x) else x.diff().dropna())
    )
    average_days = days_between.groupby('CustomerID').mean().reset_index()
    average_days.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)
    
    # Favorite shopping day of week per customer
    fav_day = (
        df_copy.groupby(['CustomerID', 'Day_Of_Week'])
        .size()
        .reset_index(name='Count')
    )
    fav_day = fav_day.loc[fav_day.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Day_Of_Week']]
    
    # Favorite shopping hour per customer
    fav_hour = (
        df_copy.groupby(['CustomerID', 'Hour'])
        .size()
        .reset_index(name='Count')
    )
    fav_hour = fav_hour.loc[fav_hour.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Hour']]
    
    # Merge all features into customer_data
    customer_data = customer_data.merge(average_days, on='CustomerID', how='left')
    customer_data = customer_data.merge(fav_day, on='CustomerID', how='left')
    customer_data = customer_data.merge(fav_hour, on='CustomerID', how='left')
    
    return customer_data


def seasonality_trends(df: pd.DataFrame, customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds seasonality and trend features based on customers' monthly spending.

    Parameters:
        df (pd.DataFrame): Transaction-level dataset with 'CustomerID', 'InvoiceDate', and 'Total_Spend'.
        customer_data (pd.DataFrame): Customer-level dataframe to enrich.

    Returns:
        pd.DataFrame: Updated customer_data with seasonality and trend features.
    """
    # Ensure InvoiceDate is datetime
    df_copy = df.copy()
    df_copy['InvoiceDate'] = pd.to_datetime(df_copy['InvoiceDate'])

    # Extract Year and Month
    df_copy['Year'] = df_copy['InvoiceDate'].dt.year
    df_copy['Month'] = df_copy['InvoiceDate'].dt.month

    # Calculate monthly spending per customer
    monthly_spending = (
        df_copy.groupby(['CustomerID', 'Year', 'Month'])['Total_Spend']
        .sum()
        .reset_index()
    )

    # Calculate mean and std deviation of monthly spending to capture seasonality
    seasonal_buying_patterns = (
        monthly_spending.groupby('CustomerID')['Total_Spend']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'Monthly_Spending_Mean', 'std': 'Monthly_Spending_Std'})
    )
    # Fill NaN std with 0 (no variability)
    seasonal_buying_patterns['Monthly_Spending_Std'] = seasonal_buying_patterns['Monthly_Spending_Std'].fillna(0)

    # Define function to calculate spending trend (slope) using linear regression
    def calculate_trend(spend_series):
        if len(spend_series) > 1:
            x = np.arange(len(spend_series))
            slope, _, _, _, _ = linregress(x, spend_series)
            return slope
        else:
            return 0

    # Calculate spending trend per customer
    spending_trends = (
        monthly_spending.groupby('CustomerID')['Total_Spend']
        .apply(calculate_trend)
        .reset_index()
        .rename(columns={'Total_Spend': 'Spending_Trend'})
    )

    # Merge new features into customer_data
    customer_data = customer_data.merge(seasonal_buying_patterns, on='CustomerID', how='left')
    customer_data = customer_data.merge(spending_trends, on='CustomerID', how='left')

    # Convert CustomerID to string type
    customer_data['CustomerID'] = customer_data['CustomerID'].astype(str)

    # Optimize data types
    customer_data = customer_data.convert_dtypes()

    return customer_data
