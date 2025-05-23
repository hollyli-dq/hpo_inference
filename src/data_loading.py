import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional

def load_sushi3_item_mapping(
    idata_path: Union[str, Path],
    item_name_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Read the *sushi3.idata* feature table and (optionally) merge in the
    item names from the ``item_mapping.txt`` list.

    Parameters
    ----------
    idata_path : str | Path
        Location of the `sushi3.idata` file that ships with the Sushi 3
        data set.  It contains **100 rows × 7 numeric attributes**.
    item_name_path : str | Path | None, optional
        Path to the *item_mapping.txt* (the 0–99 ⇒ Japanese‑Roman names
        list).  If supplied, the names are merged; otherwise the DataFrame
        is returned without the *item_name* column.

    Returns
    -------
    pandas.DataFrame
        Index : ``item_id`` (0‑99)  
        Columns (9) :

        ===========  =====================================================
        item_id      (int) repeated as a column for convenience
        item_name    (str) sushi name in Roman letters  [if mapping given]
        style        0 = maki roll | 1 = otherwise
        major_group  0 = seafood  | 1 = other
        minor_group  0–11 detailed class (see docstring/eum above)
        heaviness    0–4 0 = heavy/oily
        freq_eat     0–3 3 = user eats frequently
        price_norm   float price normalised to [0, 1]
        freq_sell    0–1 1 = most frequently sold
        ===========  =====================================================
    """
    idata_path = Path(idata_path)
    if not idata_path.exists():
        raise FileNotFoundError(f"{idata_path} not found")

    # 1) First read the raw data without enforcing types
    df = pd.read_csv(
        idata_path,
        sep=r"\s+",
        header=None,
        engine='python'  # More flexible parsing
    )
    
    # 2) Extract item_id and name from the first two columns
    item_ids = df[0].astype(int)
    item_names = df[1]
    
    # 3) Convert the remaining numeric columns
    numeric_data = df.iloc[:, 2:].astype(float)
    
    # 4) Combine into a single dataframe with proper column names
    col_names = ["style", "major_group", "minor_group",
                "heaviness", "freq_eat", "price_norm", "freq_sell"]
    
    result_df = pd.DataFrame({
        'item_id': item_ids,
        'item_name': item_names,
        **{name: numeric_data.iloc[:, i] for i, name in enumerate(col_names)}
    })
    
    # 5) Set index and convert integer columns
    result_df.set_index('item_id', inplace=True)
    
    int_cols = ["style", "major_group", "minor_group",
                "heaviness", "freq_eat", "freq_sell"]
    result_df[int_cols] = result_df[int_cols].astype(int)
    
    # 6) If item_name_path is provided, merge with that instead
    if item_name_path is not None:
        item_name_path = Path(item_name_path)
        if not item_name_path.exists():
            raise FileNotFoundError(f"{item_name_path} not found")

        # Skip header lines and parse the mapping file
        name_rows = []
        with item_name_path.open(encoding="utf-8") as fh:
            # Skip header lines that start with * or are empty
            for line in fh:
                line = line.strip()
                if not line or line.startswith('*') or line.startswith('3.'):
                    continue
                try:
                    # Try to parse as ID: Name format
                    if ':' in line:
                        _id, _rest = line.split(":", 1)
                        name_rows.append((int(_id), _rest.split("(")[0].strip()))
                except (ValueError, IndexError):
                    continue
                    
        if name_rows:  # Only proceed if we found valid name mappings
            names_df = pd.DataFrame(name_rows, columns=["item_id", "item_name"])

            # Replace the names from the data file with the ones from the mapping file
            result_df = (
                result_df
                .reset_index()
                .merge(names_df, on="item_id", how="left", suffixes=('_data', ''))
                .set_index("item_id")
            )
            
            # If we have both names, prefer the mapping file name
            if 'item_name' in result_df.columns and 'item_name_data' in result_df.columns:
                result_df['item_name'] = result_df['item_name'].fillna(result_df['item_name_data'])
                result_df = result_df.drop(columns=['item_name_data'])
            
            # Reorder columns to put item_name first
            cols = ['item_name'] + col_names
            result_df = result_df[cols]
    
    return result_df

def analyze_sushi_features(df: pd.DataFrame) -> None:
    """
    Perform comprehensive analysis of sushi features dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The sushi features dataframe from load_sushi3_item_mapping
    """
    print("\n=== Sushi Features Analysis ===\n")
    
    # 1. Basic Statistics
    print("Basic Statistics:")
    print("-----------------")
    print(df.describe())
    
    # 2. Feature Distributions
    print("\nFeature Value Counts:")
    print("--------------------")
    for col in ['style', 'major_group', 'minor_group', 'heaviness', 'freq_eat', 'freq_sell']:
        print(f"\n{col.upper()} distribution:")
        print(df[col].value_counts().sort_index())
    
    # 3. Correlation Analysis
    numeric_cols = ['style', 'major_group', 'minor_group', 'heaviness', 'freq_eat', 'price_norm', 'freq_sell']
    corr = df[numeric_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('sushi_correlations.png')
    plt.close()
    
    # 4. Category Analysis
    print("\nCategory Analysis:")
    print("-----------------")
    # Group by major_group and analyze features
    group_stats = df.groupby('major_group').agg({
        'price_norm': ['mean', 'std'],
        'heaviness': ['mean', 'std'],
        'freq_eat': ['mean', 'std'],
        'freq_sell': ['mean', 'std']
    })
    print("\nStatistics by Major Group (0=seafood, 1=other):")
    print(group_stats)
    
    # 5. Price Analysis
    print("\nPrice Analysis:")
    print("--------------")
    # Find most expensive and cheapest items
    print("\nTop 5 Most Expensive Items:")
    print(df.nlargest(5, 'price_norm')[['item_name', 'price_norm', 'freq_eat', 'freq_sell']])
    print("\nTop 5 Least Expensive Items:")
    print(df.nsmallest(5, 'price_norm')[['item_name', 'price_norm', 'freq_eat', 'freq_sell']])
    
    # 6. Popularity Analysis
    print("\nPopularity Analysis:")
    print("-------------------")
    # Combine frequency of eating and selling
    df['popularity_score'] = df['freq_eat'] + df['freq_sell']
    print("\nTop 5 Most Popular Items (based on combined freq_eat and freq_sell):")
    print(df.nlargest(5, 'popularity_score')[['item_name', 'freq_eat', 'freq_sell', 'popularity_score']])
    
    # 7. Style Analysis
    print("\nStyle Analysis:")
    print("--------------")
    style_stats = df.groupby('style').agg({
        'price_norm': ['mean', 'std'],
        'freq_eat': ['mean', 'std'],
        'freq_sell': ['mean', 'std']
    })
    print("\nStatistics by Style (0=maki roll, 1=otherwise):")
    print(style_stats)
    
    # 8. Create visualizations
    # Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='price_norm', bins=20)
    plt.title('Distribution of Sushi Prices')
    plt.xlabel('Normalized Price')
    plt.savefig('price_distribution.png')
    plt.close()
    
    # Heaviness vs Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='heaviness', y='price_norm', hue='major_group')
    plt.title('Heaviness vs Price by Major Group')
    plt.savefig('heaviness_vs_price.png')
    plt.close()
    
    print("\nAnalysis complete! Visualizations saved as:")
    print("- sushi_correlations.png")
    print("- price_distribution.png")
    print("- heaviness_vs_price.png")

def load_sushi3_user_data(udata_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and label the sushi3.udata file containing user demographic information.

    Parameters
    ----------
    udata_path : str | Path
        Path to the sushi3.udata file containing user information

    Returns
    -------
    pandas.DataFrame
        A labeled DataFrame with the following columns:
        - user_id: User identifier
        - gender: 0=male, 1=female
        - age_group: 0=15-19, 1=20-29, 2=30-39, 3=40-49, 4=50-59, 5=60+
        - survey_time: Time taken to fill questionnaire
        - childhood_prefecture: Prefecture ID where lived until age 15
        - childhood_region: Region ID where lived until age 15
        - childhood_east_west: East/West ID where lived until age 15
        - current_prefecture: Prefecture ID of current residence
        - current_region: Region ID of current residence
        - current_east_west: East/West ID of current residence
        - moved_prefecture: 0 if childhood_prefecture equals current_prefecture, 1 otherwise
    """
    # Column names for the user data
    columns = [
        'user_id',
        'gender',
        'age_group',
        'survey_time',
        'childhood_prefecture',
        'childhood_region',
        'childhood_east_west',
        'current_prefecture',
        'current_region',
        'current_east_west',
        'moved_prefecture'
    ]

    # Read the data
    udata_path = Path(udata_path)
    if not udata_path.exists():
        raise FileNotFoundError(f"{udata_path} not found")

    df = pd.read_csv(
        udata_path,
        sep='\t',
        header=None,
        names=columns
    )

    # Add categorical labels
    df['gender'] = df['gender'].map({0: 'male', 1: 'female'})
    
    age_labels = {
        0: '15-19',
        1: '20-29',
        2: '30-39',
        3: '40-49',
        4: '50-59',
        5: '60+'
    }
    df['age_group'] = df['age_group'].map(age_labels)
    
    # Convert boolean column to more descriptive values
    df['moved_prefecture'] = df['moved_prefecture'].map({
        0: 'same_prefecture',
        1: 'different_prefecture'
    })

    return df

def analyze_user_demographics(df: pd.DataFrame) -> None:
    """
    Analyze the user demographic data from the sushi dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The user demographics dataframe from load_sushi3_user_data
    """
    print("\n=== User Demographics Analysis ===\n")
    
    # 1. Basic distribution of users
    print("Gender Distribution:")
    print(df['gender'].value_counts())
    
    print("\nAge Group Distribution:")
    print(df['age_group'].value_counts().sort_index())
    
    print("\nMobility Analysis (Same vs Different Prefecture):")
    print(df['moved_prefecture'].value_counts())
    
    # 2. Create visualizations
    
    # Age distribution by gender
    plt.figure(figsize=(10, 6))
    df_grouped = df.groupby(['age_group', 'gender']).size().unstack()
    df_grouped.plot(kind='bar', stacked=True)
    plt.title('Age Distribution by Gender')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('age_gender_distribution.png')
    plt.close()
    
    # Survey completion time analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='age_group', y='survey_time')
    plt.title('Survey Completion Time by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Survey Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('survey_time_by_age.png')
    plt.close()
    
    # Regional analysis
    print("\nRegional Analysis:")
    print("\nTop 5 Childhood Regions:")
    print(df['childhood_region'].value_counts().head())
    print("\nTop 5 Current Regions:")
    print(df['current_region'].value_counts().head())
    
    # Migration patterns
    migration_patterns = pd.crosstab(
        df['childhood_region'],
        df['current_region'],
        margins=True
    )
    print("\nMigration Patterns (Region):")
    print(migration_patterns)
    
    print("\nAnalysis complete! Visualizations saved as:")
    print("- age_gender_distribution.png")
    print("- survey_time_by_age.png")

def load_sushi_rankings(order_path: Union[str, Path]) -> list:
    """
    Load sushi rankings from sushi3a.5000.10.order file.
    Each ranking is a list of 10 item IDs, where the first item is most preferred.
    
    Parameters
    ----------
    order_path : str | Path
        Path to the sushi3a.5000.10.order file
        
    Returns
    -------
    list
        List of rankings, where each ranking is a list of 10 item IDs
    """
    order_path = Path(order_path)
    if not order_path.exists():
        raise FileNotFoundError(f"{order_path} not found")
    
    rankings = []
    with open(order_path, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            # Split line and take only the item IDs (skip first two columns)
            items = line.strip().split()[2:]
            # Convert to integers
            ranking = [int(x) for x in items]
            rankings.append(ranking)
    
    return rankings

def process_sushi_covariates(features_df: pd.DataFrame) -> tuple:
    """
    Process sushi covariates, separating categorical and numerical variables.
    Categorical variables are one-hot encoded, numerical variables are normalized.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing sushi features from load_sushi3_item_mapping
        
    Returns
    -------
    tuple
        (categorical_features, numerical_features)
        - categorical_features: DataFrame with one-hot encoded categorical variables
        - numerical_features: DataFrame with normalized numerical variables
    """
    # Drop item_name as it's not a covariate
    df = features_df.copy()
    if 'item_name' in df.columns:
        df = df.drop('item_name', axis=1)
    
    # Separate categorical and numerical variables
    categorical_cols = ['style', 'major_group', 'minor_group', 'heaviness']
    numerical_cols = ['freq_eat', 'price_norm', 'freq_sell']
    
    # Add popularity_score
    df['popularity_score'] = df['freq_eat'] + df['freq_sell']
    numerical_cols.append('popularity_score')
    
    # Process categorical variables (one-hot encoding)
    categorical_features = pd.get_dummies(
        df[categorical_cols],
        columns=categorical_cols,
        prefix=categorical_cols,
        dtype=float
    )
    
    # Process numerical variables (normalization)
    numerical_features = df[numerical_cols].copy()
    for col in numerical_cols:
        # Skip if column has no variance (like freq_sell which is all zeros)
        if numerical_features[col].std() > 0:
            numerical_features[col] = (numerical_features[col] - numerical_features[col].mean()) / numerical_features[col].std()
    
    return categorical_features, numerical_features

# Example usage
if __name__ == "__main__":
    base = Path("/Users/dongqing/Desktop/research/coding/hpo_inference/data/sushi")
    
    # Load features and process covariates
    features_df = load_sushi3_item_mapping(
        idata_path     = base / "sushi3.idata",
        item_name_path = base / "item_mapping.txt"
    )
    
    cat_features, num_features = process_sushi_covariates(features_df)
    
    print("\n=== Processed Covariates ===")
    print("\nCategorical Features (first 5 rows):")
    print(cat_features.head())
    print("\nNumerical Features (first 5 rows):")
    print(num_features.head())
    print("\nFeature Dimensions:")
    print(f"Categorical features shape: {cat_features.shape}")
    print(f"Numerical features shape: {num_features.shape}")
    
    # Load rankings
    rankings = load_sushi_rankings(base / "sushi3a.5000.10.order")
    print(f"\nLoaded {len(rankings)} rankings")
