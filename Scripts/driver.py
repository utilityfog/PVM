import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def main():
    # print("This is the driver code for running and using the heart attack predictor model.")
    df = pd.read_csv('randhie.csv')
    print("ORIGINAL")
    print(df.head())
    
    print("AVERAGED")
    avg_df = average_time_series(df, "zper")
    print(avg_df.head())

    print('STANDARDIZED')
    standardized_df = standardize_df(avg_df)
    print(standardized_df.head())

    print('STANDARDIZED (CATEGORICAL VARS)')
    categorical_variables = ['plan', 'site', 'black', 'female', 'mhi', 'child', 'fchild', 'hlthg', 'hlthf', 'hlthp']
    encoded_df = encode_categorical(standardized_df, categorical_variables)
    print(encoded_df)

    print('PROCESSED')
    processed_df = replace_encoded_categorical(standardized_df, encoded_df, categorical_variables)
    print(processed_df.head())


def average_time_series(df, id_column):
    """
    A function to average attributes of a time series dataset by a unique ID.
    
    Parameters:
        df (pandas DataFrame): The input dataset.
        id_column (str): The name of the column containing unique IDs.
        attribute_columns (list): A list of column names containing attributes to be averaged.
        
    Returns:
        pandas DataFrame: A DataFrame with averaged rows, indexed by unique ID.
    """
    attributes = df.columns.tolist()
    result = df.groupby(id_column)[attributes].mean()
    return result


def standardize_df(df):
    """
    A function to standardize all the values in a DataFrame using z-score normalization,
    excluding the column containing unique identifiers (user ID column).
    
    Parameters:
        df (pandas DataFrame): The input DataFrame.
        id_column (str): The name of the column containing unique identifiers.
        
    Returns:
        pandas DataFrame: A DataFrame with standardized values.
    """
    # Copy the DataFrame excluding the ID column
    df_to_standardize = df.drop(columns=["zper"])
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the DataFrame and transform the values
    standardized_values = scaler.fit_transform(df_to_standardize.values)
    
    # Create a new DataFrame with the standardized values and the same index and columns as the original DataFrame
    standardized_df = pd.DataFrame(standardized_values, index=df.index, columns=df_to_standardize.columns)
    
    # Add back the ID column to the standardized DataFrame
    standardized_df["zper"] = df["zper"]
    
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
    # Extract categorical variables
    categorical_df = df[categorical_vars]
    
    # Perform one-hot encoding for categorical variables, drop first ensures there is no multicolinearity
    result = pd.get_dummies(categorical_df, drop_first=True)
    
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

if __name__ == "__main__":
    main()