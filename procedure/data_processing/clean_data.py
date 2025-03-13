import pandas as pd
import numpy as np

def clean_data(df):
    if df is None:
        return None

    df.dropna(subset=['Product Name'], inplace=True)

    # Handle missing values
    for col in ['Quantity Sold', 'Unit Price', 'Discount (%)', 'Total Sales', 'Profit Margin']:
        df[col] = df[col].fillna(df[col].median())

    # Outlier detection and treatment (using IQR method)
    numerical_features = ['Quantity Sold', 'Unit Price', 'Discount (%)', 'Total Sales', 'Profit Margin']
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[feature] = np.clip(df[feature], lower_bound, upper_bound)

    # Date format validation
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)

    # Remove duplicate rows
    num_duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Number of duplicate rows removed: {num_duplicates}")

    return df