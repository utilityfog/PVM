import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

hrs = pd.read_csv("life_expectancy_CleanedHRSdata.csv")
pd.set_option('display.max_columns', None)

print(f'Shape of data: {hrs.shape} \n \n Columns of data: {hrs.columns}')


### These are functions from the raw_data_processing.ipynb
def standardize_dataframe(df, numerical_columns, exclude_columns):
    """
    A function to standardize specified numerical values in a DataFrame using z-score normalization,
    excluding specified columns.
    
    Parameters:
        df (pandas DataFrame): The input DataFrame.
        numerical_columns (list): A list of numerical column names to standardize.
        exclude_columns (list): A list of column names to exclude from standardization.
        
    Returns:
        pandas DataFrame: A DataFrame with standardized numerical values.
    """
    # Exclude columns specified in exclude_columns
    columns_to_standardize = [col for col in numerical_columns if col not in exclude_columns]
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the selected columns and transform the values
    standardized_values = scaler.fit_transform(df[columns_to_standardize])
    
    # Create a new DataFrame with the standardized values and the same index and columns as the original DataFrame
    standardized_df = pd.DataFrame(standardized_values, index=df.index, columns=columns_to_standardize)
    
    # Combine the standardized numerical columns with non-numerical columns from the original DataFrame
    for col in df.columns:
        if col not in columns_to_standardize:
            standardized_df[col] = df[col]
    
    return standardized_df

def encode_categorical(df, categorical_vars):
    """
    A function to perform one-hot encoding for categorical variables in a DataFrame.
    
    Parameters:
        df (pandas DataFrame): The input DataFrame.
        categorical_columns (list): A list of column names containing categorical variables to be one-hot encoded.
        
    Returns:
        pandas DataFrame: A DataFrame with one-hot encoded categorical variables.
    """
    print(f"encodable categorical vars: {categorical_vars}")
    
    # Convert numeric categorical variables to categorical type
    for col in categorical_vars:
        df[col] = df[col].astype('category')
        
    # Extract categorical variables
    categorical_df = df[categorical_vars]
    
    # Perform one-hot encoding for categorical variables, drop first ensures there is no multicolinearity
    result = pd.get_dummies(categorical_df, dtype=float, drop_first=True)
    
    return result

def replace_encoded_categorical(df, encoded_categorical_df, categorical_columns):
    """
    A function to replace original categorical columns in a DataFrame with one-hot encoded columns.
    
    Parameters:
        df (pandas DataFrame): The original DataFrame.
        encoded_categorical_df (pandas DataFrame): The DataFrame with one-hot encoded categorical variables.
        categorical_columns (list): A list of column names containing original categorical variables.
        
    Returns:
        pandas DataFrame: A DataFrame with original categorical columns replaced by one-hot encoded columns.
    """
    # Drop original categorical columns from the original DataFrame
    df = df.drop(columns=categorical_columns)
    
    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    df = pd.concat([df, encoded_categorical_df], axis=1)
    return df

#columns_to_drop = ['Unnamed: 0', 'nt','n2','dage_y', 'rarelig']
columns_to_drop = ['Unnamed: 0', 'nt','n2','dage_y', 'rarelig']
hrs_cleaned = hrs.drop(columns=columns_to_drop, errors='ignore')
    # Replace empty strings with NaN to handle both empty strings and NaN values uniformly
hrs_cleaned = hrs_cleaned.replace('', np.nan)
    
    # Drop rows with any NaN values
hrs_cleaned = hrs_cleaned.dropna()

def is_categorical(df):
    """
    Determines which columns in a DataFrame are categorical based on data type being a string.
    
    Parameters:
        df (pandas DataFrame): The DataFrame to check.
        
    Returns:
        list: A list of column names that are considered categorical because they are of string type.
    """
    categorical_vars = [col for col in df.columns if df[col].dtype == 'object']
    return categorical_vars

# Determine categorical columns
categorical_columns = is_categorical(hrs_cleaned)  # Adjust threshold as necessary
print("Categorical columns:", categorical_columns)
# To remove columns
categorical_columns = [col for col in categorical_columns if col not in ['raedyrs']]
# Encode categorical variables
encoded_df = encode_categorical(hrs_cleaned.copy(), categorical_columns)

# Replace original categorical columns with encoded ones
final_df = replace_encoded_categorical(hrs_cleaned, encoded_df, categorical_columns)

# Convert all values to integers, setting '17.17+ yrs' specifically to 17
final_df['raedyrs'] = final_df['raedyrs'].replace('17.17+ yrs', '17').astype(int)

# Create a binary indicator for whether the education years are 17 and above
final_df['raedyrs_17plus'] = (final_df['raedyrs'] >= 17).astype(int)

# Check to make sure each "hhidpn" appears only once
def random_sample_per_group(df, group_col):
    """
    Randomly selects one observation per group from a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to sample from.
    group_col (str): The name of the column to group by.

    Returns:
    pd.DataFrame: A DataFrame with one randomly selected row per group.
    """
    # Group by the specified column and apply the sampling
    return df.groupby(group_col).apply(lambda x: x.sample(1)).reset_index(drop=True)

# Randomly sample one observation per individual
final_df = random_sample_per_group(final_df, 'hhidpn')

# Check to make sure each "hhidpn" appears only once
def check_unique_hhidpn(sampled_df, group_col):
    """
    Checks if each group identifier appears only once in the DataFrame.

    Parameters:
    sampled_df (pd.DataFrame): The DataFrame to check.
    group_col (str): The name of the column to group by.
    """
    if sampled_df[group_col].duplicated().any():
        print("Some hhidpn values appear more than once.")
    else:
        print("Each hhidpn value appears only once.")

# Perform the check
check_unique_hhidpn(final_df, 'hhidpn')

columns_to_drop = ['hhidpn', 'iwbeg', 'id']
final_df = final_df.drop(columns=columns_to_drop, errors='ignore')


# Shuffle the columns randomly
shuffled_columns = np.random.permutation(final_df.columns)

# Split the columns approximately in half
midpoint = len(shuffled_columns) // 2
first_half_columns = shuffled_columns[:midpoint]
second_half_columns = shuffled_columns[midpoint:]

# Create two new DataFrames based on these split columns
df_first_half = final_df[first_half_columns]
df_second_half = final_df[second_half_columns]

# Print out the columns to verify
print("First half columns:", df_first_half.columns)
print("Second half columns:", df_second_half.columns)