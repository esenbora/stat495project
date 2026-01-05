import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
DATA_PATH = '/Users/boraesen/Desktop/stat495project/data/raw/afad_full_historical_1990_2025.csv'
OUTPUT_DIR = '/Users/boraesen/Desktop/stat495project/data/processed/eda_plots'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    """Loads the dataset and performs initial cleaning."""
    print(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Cleans the dataset."""
    print("Cleaning data...")
    # Convert Date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with missing critical values if necessary (optional for now)
    # df.dropna(subset=['magnitude', 'depth', 'latitude', 'longitude'], inplace=True)
    
    print("Data cleaning complete.")
    return df

def perform_univariate_analysis(df):
    """Performs univariate analysis and saves plots."""
    print("Performing univariate analysis...")
    
    # Magnitude Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['magnitude'], bins=30, kde=True)
    plt.title('Distribution of Earthquake Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, 'magnitude_distribution.png'))
    plt.close()
    
    # Depth Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['depth'], bins=30, kde=True)
    plt.title('Distribution of Earthquake Depths')
    plt.xlabel('Depth (km)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, 'depth_distribution.png'))
    plt.close()

    # Temporal Distribution (Yearly)
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        plt.figure(figsize=(12, 6))
        sns.countplot(x='year', data=df)
        plt.title('Number of Earthquakes per Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(OUTPUT_DIR, 'yearly_distribution.png'))
        plt.close()

def perform_multivariate_analysis(df):
    """Performs multivariate analysis and saves plots."""
    print("Performing multivariate analysis...")
    
    # Magnitude vs Depth
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='depth', y='magnitude', data=df, alpha=0.5)
    plt.title('Magnitude vs Depth')
    plt.xlabel('Depth (km)')
    plt.ylabel('Magnitude')
    plt.savefig(os.path.join(OUTPUT_DIR, 'magnitude_vs_depth.png'))
    plt.close()
    
    # Spatial Distribution
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='longitude', y='latitude', data=df, hue='magnitude', palette='viridis', alpha=0.5, size='magnitude', sizes=(10, 200))
    plt.title('Spatial Distribution of Earthquakes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Magnitude', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_distribution.png'))
    plt.close()

def save_summary_stats_as_image(df):
    """Saves summary statistics as an image."""
    print("Saving summary statistics as image...")
    
    # Select relevant numeric columns
    cols_to_analyze = ['magnitude', 'depth', 'latitude', 'longitude']
    # Filter to existing columns
    cols = [c for c in cols_to_analyze if c in df.columns]
    
    if not cols:
        print("No numeric columns found for summary statistics.")
        return

    stats_df = df[cols].describe().T
    
    # Add Skewness and Kurtosis
    stats_df['skew'] = df[cols].skew()
    stats_df['kurtosis'] = df[cols].kurt()
    
    # Reorder columns to put count first, then mean, std, min, percentiles, max, skew, kurt
    columns_order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']
    stats_df = stats_df[columns_order]
    
    # Round values
    stats_df = stats_df.round(2)
    
    plt.figure(figsize=(14, 6)) # Increased width for more columns
    plt.axis('off')
    
    # Create table
    table = plt.table(cellText=stats_df.values, 
                      colLabels=stats_df.columns, 
                      rowLabels=stats_df.index, 
                      loc='center', 
                      cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5) # Scale up a bit more
    
    plt.title('Enhanced Summary Statistics', y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_statistics.png'), bbox_inches='tight', dpi=300)
    plt.close()

def save_data_info_as_image(df):
    """Saves data info (dtypes, non-null counts) as an image."""
    print("Saving data info as image...")
    
    # Create a DataFrame for info
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count().values,
        'Dtype': df.dtypes.values
    })
    
    plt.figure(figsize=(10, len(df.columns) * 0.5 + 1))
    plt.axis('off')
    
    table = plt.table(cellText=info_df.values, 
                      colLabels=info_df.columns, 
                      loc='center', 
                      cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Data Information', y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_info.png'), bbox_inches='tight', dpi=300)
    plt.close()

def save_head_as_image(df):
    """Saves the first few rows of the dataset as an image."""
    print("Saving data head as image...")
    
    head_df = df.head()
    
    plt.figure(figsize=(14, 4))
    plt.axis('off')
    
    table = plt.table(cellText=head_df.values, 
                      colLabels=head_df.columns, 
                      loc='center', 
                      cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(8) # Smaller font for head as it has many columns
    table.scale(1.2, 1.5)
    
    plt.title('First 5 Rows', y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_head.png'), bbox_inches='tight', dpi=300)
    plt.close()

def perform_correlation_analysis(df):
    """Performs correlation analysis and saves heatmap."""
    print("Performing correlation analysis...")
    
    # Select relevant numeric columns
    cols_to_analyze = ['magnitude', 'depth', 'latitude', 'longitude']
    # Filter to existing columns
    cols = [c for c in cols_to_analyze if c in df.columns]
    
    if len(cols) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return

    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    df = load_data(DATA_PATH)
    if df is not None:
        # Basic Info
        print("\nDataset Info:")
        print(df.info())
        print("\nSummary Statistics:")
        print(df.describe())
        
        df = clean_data(df)
        
        save_data_info_as_image(df)
        save_head_as_image(df)
        save_summary_stats_as_image(df)
        perform_univariate_analysis(df)
        perform_multivariate_analysis(df)
        perform_correlation_analysis(df)
        
        print(f"\nEDA complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
