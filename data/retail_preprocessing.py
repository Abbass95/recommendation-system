
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load data from xlsx file."""
    return pd.read_excel(path)

def data_description(df):
    """
    Print basic information and statistics about the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to describe.

    Returns:
        dict: A dictionary containing head, info, numerical and categorical descriptions.
    """
    print("🔹 First 5 rows:")
    print(df.head(), "\n")

    print("🔹 DataFrame Info:")
    df_info = df.info()
    print()

    print("🔹 Summary Statistics (Numerical):")
    numeric_desc = df.describe().T
    print(numeric_desc, "\n")

    print("🔹 Summary Statistics (Categorical):")
    categorical_desc = df.describe(include='object').T
    print(categorical_desc, "\n")

    return 

def handle_missing_values(df, plot=True, drop_columns=['CustomerID', 'Description']):
    """
    Handle and visualize missing values in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        plot (bool): Whether to plot the missing value percentages.
        drop_columns (list): List of columns to drop rows from if they contain missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame with selected missing rows dropped.
    """
    # Calculate missing data percentages
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data[missing_data > 0] / len(df)) * 100
    missing_percentage.sort_values(ascending=True, inplace=True)

    # Plot missing values if requested
    if plot and not missing_percentage.empty:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.barh(missing_percentage.index, missing_percentage, color="#ff6200f7")

        for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
            ax.text(value + 0.5, i, f"{value:.2f}%", ha='left', va='center',
                    fontweight='bold', fontsize=14, color='black')

        ax.set_xlim([0, max(40, missing_percentage.max() + 5)])
        plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
        plt.xlabel("Percentages (%)", fontsize=16)
        plt.tight_layout()
        plt.show()

    # Drop rows with missing values in specified columns
    df_cleaned = df.dropna(subset=drop_columns)

    # Print total remaining missing values
    total_missing_after = df_cleaned.isnull().sum().sum()
    print(f"✅ Remaining total missing values: {total_missing_after}")

    return df_cleaned

def handle_duplicates(df, inspect=False, sort_by=None):
    """
    Detect and remove duplicate rows from the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        inspect (bool): Whether to return the duplicate rows before removal for inspection.
        sort_by (list or None): Columns to sort duplicates by when inspecting.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
        pd.DataFrame (optional): Duplicate rows before removal (if inspect=True).
    """
    num_duplicates = df.duplicated().sum()
    print(f"🔁 The dataset contains {num_duplicates} duplicate rows.")

    if inspect and num_duplicates > 0:
        duplicate_rows = df[df.duplicated(keep=False)]
        if sort_by:
            duplicate_rows = duplicate_rows.sort_values(by=sort_by)
        else:
            duplicate_rows = duplicate_rows.sort_index()
        print("🔍 Showing first 10 duplicate rows:")
        print(duplicate_rows.head(10))

    # Drop duplicates and return a new DataFrame
    df_cleaned = df.drop_duplicates().copy()
    print("✅ Duplicates removed.")

    if inspect and num_duplicates > 0:
        return df_cleaned, duplicate_rows
    return df_cleaned

def remove_canceled_baskets(df):
    """
    Remove canceled transactions from the dataset.
    Canceled transactions are identified by InvoiceNo starting with 'C'.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing an 'InvoiceNo' column.

    Returns:
        pd.DataFrame: DataFrame without canceled transactions.
    """
    initial_shape = df.shape
    df_cleaned = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
    removed = initial_shape[0] - df_cleaned.shape[0]
    print(f"🗑️ Removed {removed} canceled transactions (InvoiceNo starting with 'C').")
    
    return df_cleaned

def analyze_stockcode_anomalies(df, plot=True, remove_anomalies=True):
    """
    Analyze stock code anomalies based on numeric character content and frequency.
    Optionally remove anomalous stock codes (with 0 or 1 numeric characters).

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'StockCode' column.
        plot (bool): Whether to plot the top 10 stock codes.
        remove_anomalies (bool): Whether to remove rows with anomalous stock codes.

    Returns:
        pd.DataFrame: Cleaned DataFrame (if `remove_anomalies=True`), otherwise the original.
    """
    # 1. Unique stock codes
    num_unique = df['StockCode'].nunique()
    print(f"📦 Unique stock codes: {num_unique}")

    # 2. Top 10 frequent stock codes
    top_10 = df['StockCode'].value_counts(normalize=True).head(10) * 100

    if plot:
        plt.figure(figsize=(12, 5))
        top_10.plot(kind='barh', color='#ff6200')
        for idx, val in enumerate(top_10):
            plt.text(val, idx + 0.25, f'{val:.2f}%', fontsize=10)
        plt.title('Top 10 Most Frequent Stock Codes')
        plt.xlabel('Percentage Frequency (%)')
        plt.ylabel('Stock Codes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # 3. Numeric character distribution in stock codes
    unique_codes = df['StockCode'].unique()
    numeric_counts = pd.Series(unique_codes).apply(lambda x: sum(c.isdigit() for c in str(x)))
    count_distribution = numeric_counts.value_counts()
    print("🔢 Numeric character count distribution in unique stock codes:")
    print("-" * 60)
    print(count_distribution)

    # 4. Identify anomalous codes (0 or 1 digits)
    anomalous_codes = [code for code in unique_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]
    print("\n🚨 Anomalous stock codes (0 or 1 digits):")
    print("-" * 30)
    for code in anomalous_codes:
        print(code)

    # 5. Percentage of records with anomalous stock codes
    perc_anomalous = (df['StockCode'].isin(anomalous_codes).sum() / len(df)) * 100
    print(f"\n📊 {perc_anomalous:.2f}% of records contain anomalous stock codes.")

    # 6. Remove if specified
    if remove_anomalies:
        df = df[~df['StockCode'].isin(anomalous_codes)].copy()
        print("✅ Rows with anomalous stock codes removed.")

    return df

def clean_description(df, plot_top=True, remove_service_related=True):
    """
    Cleans and standardizes the 'Description' column in the dataset.
    
    Steps:
    - Plot the top 30 most frequent product descriptions (optional).
    - Detect descriptions containing lowercase characters.
    - Remove service-related descriptions (optional).
    - Standardize descriptions to uppercase.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a 'Description' column.
        plot_top (bool): Whether to plot the top 30 most frequent descriptions.
        remove_service_related (bool): Whether to remove service-related entries.

    Returns:
        pd.DataFrame: A cleaned copy of the original DataFrame.
    """
    # 1. Plot top 30 frequent descriptions
    if plot_top:
        desc_counts = df['Description'].value_counts().head(30)
        plt.figure(figsize=(12, 8))
        plt.barh(desc_counts.index[::-1], desc_counts.values[::-1], color='#ff6200')
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Description')
        plt.title('Top 30 Most Frequent Descriptions')
        plt.tight_layout()
        plt.show()

    # 2. Print descriptions containing lowercase letters
    lowercase_descriptions = [desc for desc in df['Description'].unique() if any(char.islower() for char in str(desc))]
    print("🔎 Unique descriptions containing lowercase characters:")
    print("-" * 60)
    for desc in lowercase_descriptions:
        print(desc)

    # 3. Handle service-related descriptions
    service_related = ["Next Day Carriage", "High Resolution Image"]
    service_pct = df[df['Description'].isin(service_related)].shape[0] / df.shape[0] * 100
    print(f"\n📦 Service-related entries represent {service_pct:.2f}% of the dataset.")

    df_cleaned = df.copy()
    if remove_service_related:
        df_cleaned = df_cleaned[~df_cleaned['Description'].isin(service_related)]
        print("✅ Removed rows with service-related descriptions.")

    # 4. Standardize descriptions to uppercase
    df_cleaned['Description'] = df_cleaned['Description'].str.upper()

    return df_cleaned

def treat_zero_price(df, verbose=True):
    """
    Removes records with UnitPrice equal to 0, which may indicate data entry errors.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'UnitPrice' column.
        verbose (bool): Whether to print summary stats and logs.

    Returns:
        pd.DataFrame: Cleaned DataFrame with zero-priced records removed.
    """
    if verbose:
        zero_price_df = df[df['UnitPrice'] == 0]
        count = len(zero_price_df)
        if count > 0:
            print(f"⚠️ Found {count} records with UnitPrice = 0.")
            print("🔍 Summary of Quantity for zero-priced items:")
            print(zero_price_df['Quantity'].describe())
        else:
            print("✅ No records with UnitPrice = 0 found.")
    
    df_cleaned = df[df['UnitPrice'] > 0].copy()

    if verbose and count > 0:
        print(f"✅ Removed {count} zero-priced records.")

    return df_cleaned

def leroy_merlin_data_adaptation(df, verbose=True):
    """
    Adapts raw transaction data to fit Leroy Merlin-style product segmentation:
    - Converts InvoiceDate to datetime
    - Creates BasketID and ScanID
    - Standardizes and filters descriptions by relevant DIY/home keywords

    Parameters:
        df (pd.DataFrame): Original e-commerce DataFrame with required columns.
        verbose (bool): Print status messages and counts if True.

    Returns:
        pd.DataFrame: Filtered and adapted DataFrame for analysis.
    """
    required_cols = ['InvoiceDate', 'InvoiceNo', 'CustomerID', 'Description']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Convert to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create BasketID
    df['BasketID'] = df['InvoiceNo'].astype(str)

    # Standardize product descriptions
    df['Description'] = df['Description'].str.strip().str.lower()

    # Define relevant category keywords
    category_keywords = {
        'Home Decor': ['lantern', 'vase', 'cushion', 'lamp', 'mirror', 'frame', 'canvas', 'rug', 'curtain', 'blind'],
        'Storage & Organization': ['box', 'drawer', 'storage', 'basket', 'shelf', 'rack', 'wardrobe', 'cabinet'],
        'DIY Hardware': ['hook', 'holder', 'hinge', 'bracket', 'screw', 'nail', 'bolt', 'plug'],
        'Lighting': ['lamp', 'bulb', 'ceiling light', 'spotlight', 'chandelier', 'wall light'],
        'Garden': ['pot', 'planter', 'hose', 'spade', 'rake', 'watering can', 'greenhouse'],
        'Bathroom': ['mirror', 'soap dish', 'towel rail', 'toilet brush', 'bath mat'],
        'Kitchen': ['utensil holder', 'cutlery tray', 'storage jar', 'rack', 'spice rack'],
    }

    all_keywords = set(kw for kws in category_keywords.values() for kw in kws)
    keyword_pattern = '|'.join([r'\b' + kw + r'\b' for kw in all_keywords])

    df_filtered = df[df['Description'].str.contains(keyword_pattern, na=False)]

    # Create ScanID (as if customer scanned loyalty card)
    df_filtered['ScanID'] = df_filtered['CustomerID'].astype(str) + "_" + df_filtered['BasketID']

    if verbose:
        print(f"✅ Adapted data contains {len(df_filtered)} records out of {len(df)} total.")
        print(f"📦 {df_filtered['ScanID'].nunique()} unique customer-basket scans.")

    return df_filtered

def data_split(df_diy, cutoff_date="2011-09-01", verbose=True):
    """
    Splits the input DataFrame into training and future (holdout/test) sets based on InvoiceDate.

    Parameters:
        df_diy (pd.DataFrame): Preprocessed DataFrame including 'InvoiceDate'.
        cutoff_date (str or pd.Timestamp): Date used to split the data. Defaults to '2011-09-01'.
        verbose (bool): Whether to print the size of each split.

    Returns:
        tuple: (df_train, df_future)
    """
    if 'InvoiceDate' not in df_diy.columns:
        raise ValueError("Input DataFrame must contain 'InvoiceDate' column.")
    
    cutoff = pd.to_datetime(cutoff_date)

    # Perform the split
    df_future = df_diy[df_diy['InvoiceDate'] > cutoff].copy()
    df_train = df_diy[df_diy['InvoiceDate'] <= cutoff].copy()

    if verbose:
        print(f"🧪 Future set size: {len(df_future)}")
        print(f"📚 Training set size: {len(df_train)}")

    return df_train, df_future

def outlier_detection(customer_data: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detects and removes outliers from customer data using Isolation Forest.

    Parameters:
        customer_data (pd.DataFrame): The input dataframe containing customer features.
        contamination (float): The proportion of outliers in the data. Default is 0.05 (5%).

    Returns:
        pd.DataFrame: Cleaned customer data with outliers removed.
    """
    # Fit Isolation Forest (excluding 'CustomerID')
    model = IsolationForest(contamination=contamination, random_state=0)
    features = customer_data.drop(columns='CustomerID')
    customer_data['Outlier_Scores'] = model.fit_predict(features.to_numpy())

    # Flag outliers: 1 = inlier, -1 = outlier => convert to binary flag
    customer_data['Is_Outlier'] = (customer_data['Outlier_Scores'] == -1).astype(int)

    # Plot percentage of inliers and outliers
    outlier_percentage = customer_data['Is_Outlier'].value_counts(normalize=True) * 100
    labels = {0: 'Inliers', 1: 'Outliers'}

    plt.figure(figsize=(10, 4))
    outlier_percentage.rename(index=labels).plot(kind='barh', color=['#4caf50', '#f44336'])
    for index, value in enumerate(outlier_percentage):
        plt.text(value + 1, index, f'{value:.2f}%', fontsize=12)
    plt.title('Percentage of Inliers and Outliers')
    plt.xlabel('Percentage')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()

    # Remove outliers
    customer_data_cleaned = customer_data[customer_data['Is_Outlier'] == 0].drop(columns=['Outlier_Scores', 'Is_Outlier']).reset_index(drop=True)

    return customer_data_cleaned

def correlation_analysis(customer_data_cleaned: pd.DataFrame):
    """
    Displays a heatmap of the correlation matrix for the cleaned customer data.

    Parameters:
        customer_data_cleaned (pd.DataFrame): The dataframe with numerical features (excluding outliers).
    """
    # Ensure 'CustomerID' is excluded from correlation analysis
    if 'CustomerID' in customer_data_cleaned.columns:
        data = customer_data_cleaned.drop(columns=['CustomerID'])
    else:
        data = customer_data_cleaned.copy()

    # Compute correlation matrix
    corr = data.corr()

    # Define a custom diverging colormap
    colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']
    cmap = LinearSegmentedColormap.from_list('custom_map', colors, N=256)

    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set plot style
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 10))

    # Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt='.2f',
        center=0,
        linewidths=2,
        square=True,
        cbar_kws={"shrink": .8}
    )

    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numerical features of the given DataFrame using StandardScaler.

    Parameters:
        df (pd.DataFrame): The customer dataset with numerical features.

    Returns:
        pd.DataFrame: A scaled version of the customer data.
    """
    # Columns to exclude from scaling
    columns_to_exclude = ['CustomerID', 'Day_Of_Week']

    # Select columns to scale: only numeric and not excluded
    columns_to_scale = df.select_dtypes(include=[np.number]).columns.difference(columns_to_exclude)

    # Initialize the scaler
    scaler = StandardScaler()

    # Copy original data
    df_scaled = df.copy()

    # Apply scaling
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    return df_scaled   