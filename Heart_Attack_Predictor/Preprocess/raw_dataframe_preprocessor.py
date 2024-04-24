import os
from typing import List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset

from .globals import RANDHIE_CATEGORICAL_VARIABLES, RANDHIE_NUMERIC_VARIABLES, HEART_CATEGORICAL_VARIABLES, HEART_NUMERIC_VARIABLES

FINAL_RANDHIE_REGRESSORS = []
FINAL_RANDHIE_Y = []

FINAL_HEART_REGRESSORS = []
FINAL_HEART_Y = []

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
    result = pd.get_dummies(categorical_df, dtype=float, drop_first=False)
    
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

def remove_nan_or_inf(df: pd.DataFrame):
    """
    Removes rows from a DataFrame that contain NaN or infinite values in any column.

    Parameters:
    - df: pd.DataFrame

    Returns:
    - pd.DataFrame: DataFrame with rows containing NaN or infinite values removed.
    """
    # Remove rows with NaN values
    df_cleaned = df.dropna()

    # Remove rows with infinite values
    df_cleaned = df_cleaned[~(df_cleaned.isin([np.inf, -np.inf]).any(axis=1))]

    return df_cleaned

def save_dataframe(df, directory, filename):
    """
    Save a DataFrame to a specified directory as a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        directory (str): The directory where the CSV file will be saved.
        filename (str): The filename to use for the saved CSV file.

    Returns:
        None
    """
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Ensure the filename ends with '.csv'
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Define the full path where the file will be saved
    file_path = os.path.join(directory, filename)

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved successfully to {file_path}")

class RANDHIE:
    def original_preprocess(self, df_path):
        """
        Method that pre-processes the randhie.csv dataset exactly as it was preprocessed in the demand for medical care paper
        
        Returns: pd.DataFrame
        """
        df = pd.read_csv(df_path)
        print(f"Raw DataFrame: {df.head()}")

        # Re-Constructing Variables for the Four Equations from the Paper: ""
        df['positive_med_exp'] = (df['meddol'] > 0).astype(int)  # 1 if positive medical expenses, else 0
        df['positive_inpatient_exp'] = ((df['inpdol'] > 0) & (df['meddol'] > 0)).astype(int)  # 1 if positive inpatient expenses and positive medical use, else 0
        df['only_outpatient_exp'] = ((df['inpdol'] == 0) & (df['meddol'] > 0)).astype(int)  # 1 if only outpatient expenses and positive medical use, else 0
        df['log_med_exp'] = np.where(df['meddol'] > 0, np.log(df['meddol']), 0)  # Log transformation for positive expenses
        df['log_inpatient_exp'] = np.where(df['inpdol'] > 0, np.log(df['inpdol']), 0)  # Log transformation for positive inpatient expenses

        # Define independent variables based on the paper's model and available data
        X_vars = ['xage', 'linc', 'coins', 'black', 'female', 'educdec']
        X = df[X_vars]
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        # Equation 1: Probit model for zero versus positive medical expenses
        model_1 = Probit(df['positive_med_exp'], X).fit()

        # Equation 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services
        df_pos_med_exp = df[df['positive_med_exp'] == 1]  # Filter for positive medical use
        model_2 = Probit(df_pos_med_exp['positive_inpatient_exp'], X.loc[df_pos_med_exp.index]).fit()

        # Equation 3: OLS regression for log of positive medical expenses if only outpatient services are used
        df_only_outpatient_exp = df[df['only_outpatient_exp'] == 1]
        model_3 = OLS(df_only_outpatient_exp['log_med_exp'], X.loc[df_only_outpatient_exp.index]).fit()

        # Equation 4: OLS regression for log of medical expenses for those with any inpatient expenses
        df_pos_inpatient_exp = df[df['positive_inpatient_exp'] == 1]
        model_4 = OLS(df_pos_inpatient_exp['log_inpatient_exp'], X.loc[df_pos_inpatient_exp.index]).fit()

        # Print summaries of the models
        print("Model 1: Probit model for zero versus positive medical expenses")
        print(model_1.summary())
        print("\nModel 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services")
        print(model_2.summary())
        print("\nModel 3: OLS regression for log of positive medical expenses if only outpatient services are used")
        print(model_3.summary())
        print("\nModel 4: OLS regression for log of medical expenses for those with any inpatient expenses")
        print(model_4.summary())
        
    def improved_preprocess(self, df_path):
        """
        Method that pre-processes the randhie.csv dataset such that we can test at the individual patient level which predictors are statistically significant
        
        Returns: pd.DataFrame
        """
        # Read
        df = pd.read_csv(df_path)
        print(f"Raw DataFrame: {df.head()}")
        
        df.drop(['rownames'], axis=1, inplace=True)
        
        # Remove rows with NaN or inf values
        df_cleaned = remove_nan_or_inf(df)
        print(f"DataFrame after removing NaN and inf values: {df_cleaned.head()}")
        # Test 0
        save_dataframe(df_cleaned, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "randhie_preprocessed0.csv")
        
        # Average the numeric column values at the patient (zper) level since we are not interested in time and the RANDHIE experiment has 5 individual year observations for each patient
        collapsed_df = self.average_by_unique_patient(df_cleaned, "zper", RANDHIE_CATEGORICAL_VARIABLES)
        print(f"AVERAGED: {collapsed_df.head()}")
        # Test 1
        save_dataframe(collapsed_df, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "randhie_preprocessed1.csv")
        
        # Re-Constructing Variables for the Four Equations from the Paper (must be done before standardization as it is affected by sign): ""
        collapsed_df['is_positive_med_exp'] = (collapsed_df['meddol'] > 0).astype(int)  # 1 if positive medical expenses, else 0
        collapsed_df['is_positive_inpatient_exp'] = ((collapsed_df['inpdol'] > 0) & (collapsed_df['meddol'] > 0)).astype(int)  # 1 if positive inpatient expenses and positive medical use, else 0
        collapsed_df['is_only_outpatient_exp'] = ((collapsed_df['inpdol'] == 0) & (collapsed_df['meddol'] > 0)).astype(int)  # 1 if only outpatient expenses and positive medical use, else 0
        collapsed_df['log_med_exp'] = np.where(collapsed_df['meddol'] > 0, np.log(collapsed_df['meddol']), 0)  # Log transformation for positive expenses
        collapsed_df['log_inpatient_exp'] = np.where(collapsed_df['inpdol'] > 0, np.log(collapsed_df['inpdol']), 0)  # Log transformation for positive inpatient expenses
        paper_variables_categorical = ['is_positive_med_exp', 'is_positive_inpatient_exp', 'is_only_outpatient_exp']
        paper_variables_numeric = ['log_med_exp', 'log_inpatient_exp']
        print(f"processed_df's generated columns: {collapsed_df[paper_variables_numeric + paper_variables_categorical].head()}")
        # Test 2
        save_dataframe(collapsed_df, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "randhie_preprocessed2.csv")
        
        # Standardize Numeric Columns: standardized_df = standardize_df(avg_df)
        new_numeric_columns = RANDHIE_NUMERIC_VARIABLES + paper_variables_numeric
        new_categorical_columns = RANDHIE_CATEGORICAL_VARIABLES + paper_variables_categorical
        standardized_df = standardize_dataframe(collapsed_df, new_numeric_columns, new_categorical_columns)
        print(f"STANDARDIZED - NEW: {standardized_df.head()}")
        # Add plan without standardization
        standardized_df['plan'] = collapsed_df['plan']
        # Test 3
        save_dataframe(standardized_df, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "randhie_preprocessed3.csv")
        
        # Combine standardized_df's child and fchild into one categorical column as they are redundant
        # Create a new categorical column combining 'child' and 'fchild'
        conditions = [
            (standardized_df['child'] == 1) & (standardized_df['fchild'] == 0),
            (standardized_df['child'] == 1) & (standardized_df['fchild'] == 1),
            (standardized_df['child'] == 0)
        ]
        choices = ['mchild', 'fchild', 'adult']
        standardized_df['person_type'] = np.select(conditions, choices, default='adult')
        # Append newly constructed person_type
        new_categorical_columns.append('person_type')
        
        # Drop the original 'child' and 'fchild' columns as no longer needed
        standardized_df.drop(['child', 'fchild'], axis=1, inplace=True)
        for item in ['child', 'fchild']:
            new_categorical_columns.remove(item)
        # Test 4
        save_dataframe(standardized_df, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "randhie_preprocessed4.csv")
        
        # One Hot Encoding categorical variables
        encoded_df = encode_categorical(standardized_df, new_categorical_columns)
        print(f"ONE HOT ENCODED (CATEGORICAL VARS): {encoded_df.head()}")
        
        # Replacing the randhie df's categorical variables with one hot encoded variables
        processed_df = replace_encoded_categorical(standardized_df, encoded_df, new_categorical_columns)
        print(f"PROCESSED: {processed_df.head()}")
        
        # Check if final preprocessing has been done correctly
        save_dataframe(processed_df, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "randhie_preprocessed_final.csv")

        # Define independent variables based on the paper's model and available data: 
            # Excluding variables that are known to be endogenous (e.g. if someone makes a lot of hospital visits, obviously it will have a positive causal relationship with their quantity demanded for medical care even if there are confounding variables that may affect the number of times they visit the hospital)
            # Excluding variables that are a deterministic function of another to prevent perfect multicolinearity; no inclusion of both linc and income
            # We use one hot encoding instead of dummy encoding because vectorization is affected by that choice. We want to capture maximum information for each vectorized row
        X_list = ['person_type_adult', 'person_type_fchild', 'person_type_mchild', 'hlthg_0', 'hlthg_1', 'hlthf_0',	'hlthf_1', 'hlthp_0', 'hlthp_1', 'female_0', 'female_1', 'site_2', 'site_3', 'site_4', 'site_5', 'site_6', 'tookphys_0', 'tookphys_1', 'plan', 'xage', 'educdec', 'time', 'disea', 'physlm', 'mdeoff', 'lfam', 'lpi', 'logc', 'xghindx', 'linc', 'lnum', 'black', 'mhi']
        X = processed_df[X_list]
        # print(f"randhie final X: {X}") # The ordering of the columns specified in X_list must be preserved!!
        X = sm.add_constant(X)  # Adds an intercept term
        
        # Store both final X_list (order preserved) and final y variables in global lists
        FINAL_RANDHIE_REGRESSORS = X_list
        FINAL_RANDHIE_Y = ['is_positive_med_exp_1', 'is_positive_inpatient_exp_1', 'log_med_exp', 'log_inpatient_exp']
        
        # FOUR EQUATION MODEL ACCORDING TO PAPER: Health Insurance and the Demand for Medical Care

        # Equation 1: Probit model for zero versus positive medical expenses
        model_1 = Probit(processed_df['is_positive_med_exp_1'], X).fit() # LASSO

        # Equation 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services
        df_pos_med_exp = processed_df[processed_df['is_positive_med_exp_1'] == 1]  # Filter for positive medical use
        model_2 = Probit(df_pos_med_exp['is_positive_inpatient_exp_1'], X.loc[df_pos_med_exp.index]).fit() # LASSO

        # Equation 3: OLS regression for log of positive medical expenses if only outpatient services are used
        df_only_outpatient_exp = processed_df[processed_df['is_only_outpatient_exp_1'] == 1]
        model_3 = OLS(df_only_outpatient_exp['log_med_exp'], X.loc[df_only_outpatient_exp.index]).fit()

        # Equation 4: OLS regression for log of medical expenses for those with any inpatient expenses
        df_pos_inpatient_exp = processed_df[processed_df['is_positive_inpatient_exp_1'] == 1]
        model_4 = OLS(df_pos_inpatient_exp['log_inpatient_exp'], X.loc[df_pos_inpatient_exp.index]).fit()

        # Print summaries of the models
        print("Model 1: Probit model for zero versus positive medical expenses")
        print(model_1.summary())
        print("\nModel 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services")
        print(model_2.summary())
        print("\nModel 3: OLS regression for log of positive medical expenses if only outpatient services are used")
        print(model_3.summary())
        print("\nModel 4: OLS regression for log of medical expenses for those with any inpatient expenses")
        print(model_4.summary())
        
        return processed_df
    
    def average_by_unique_patient(self, df: pd.DataFrame, id_column: str, categorical_cols=[]):
        """
        A function to average attributes of a time series dataset by a unique ID, 
        averaging true numeric columns and randomly selecting values for categorical columns.
        
        Parameters:
            df (pandas DataFrame): The input dataset.
            id_column (str): The name of the column containing unique IDs.
            categorical_cols (list): A list of column names that are categorical but represented numerically.
            
        Returns:
            pandas DataFrame: A DataFrame with averaged numeric columns and randomly selected categorical columns, indexed by unique ID.
        """
        # Identify numeric columns (excluding categorical ones)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in categorical_cols and col != id_column]
        
        # Non-numeric columns are either categorical or explicitly excluded from numeric
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist() + categorical_cols
        non_numeric_cols = [col for col in non_numeric_cols if col != id_column]
        
        # Group by ID and compute the mean for numeric columns
        if numeric_cols:
            df_numeric_mean = df.groupby(id_column)[numeric_cols].mean()
        else:
            df_numeric_mean = pd.DataFrame(index=df[id_column].unique())

        # Group by ID and randomly select a value for non-numeric columns
        df_non_numeric_random = df.groupby(id_column)[non_numeric_cols].agg(lambda x: np.random.choice(x.dropna()))

        # Combine the results
        result = pd.concat([df_numeric_mean, df_non_numeric_random], axis=1)
        
        # Reset the index to make 'id_column' a regular column again
        result.reset_index(inplace=True)
        
        return result
    
class HEART:
    def preprocess(self, df_path):
        """
        Method that pre-processes the heart_attack_prediction.csv dataset
        
        CleaningDecisions:
        1. Seperate Systolic and Diastolic blood pressure into seperate columns 
        2. Create an interaction term between systolic and diastolic.
        3. Drop 'Sex_Male', 'Diet_Average', 'Country_Argentina' to prevent multicolinearity.
        4. Set categorical variables to category type.
        
        Returns: pd.DataFrame
        """
        # Read
        df = pd.read_csv(df_path)
        print(f"Raw DataFrame: {df.head()}")
        
        # Remove rows with NaN or inf values
        df_cleaned = remove_nan_or_inf(df)
        print(f"DataFrame after removing NaN and inf values: {df_cleaned.head()}")
        # Test 0 
        save_dataframe(df_cleaned, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "heart_preprocessed0.csv")
        
        # Get Log Income
        df_cleaned['Log_Income'] = np.log(df_cleaned['Income'])
        
        # Standardize Numeric Columns: standardized_df = standardize_df(avg_df)
        numeric_columns = HEART_NUMERIC_VARIABLES
        categorical_columns = HEART_CATEGORICAL_VARIABLES
        standardized_df = standardize_dataframe(df_cleaned, numeric_columns, categorical_columns)
        print(f"STANDARDIZED - NEW: {standardized_df.head()}")
        # Test 1
        save_dataframe(standardized_df, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "heart_preprocessed1.csv")
        
        # One Hot Encoding categorical variables
        encoded_df = encode_categorical(standardized_df, categorical_columns)
        print(f"ONE HOT ENCODED (CATEGORICAL VARS): {encoded_df.head()}")
        
        # Replacing the heart df's categorical variables with one hot encoded variables
        processed_df = replace_encoded_categorical(standardized_df, encoded_df, categorical_columns)
        print(f"PROCESSED: {processed_df.head()}")
        
        ### Seperate systolic and diastolic blood pressure into their own variables and create an interaction term.
        # Split the 'Blood Pressure' column into 'Systolic' and 'Diastolic'
        processed_df[['Systolic', 'Diastolic']] = processed_df['Blood Pressure'].str.split('/', expand=True).astype(int)
        
        # Create interaction term by multiplying Systolic and Diastolic pressures
        processed_df['BP_Interaction'] = processed_df['Systolic'] * processed_df['Diastolic']
        
        # Drop unnecessary variables
        processed_df = processed_df.drop(['Patient ID', 'Blood Pressure', 'Income', 'Continent', 'Hemisphere'], axis=1)

        # Define independent variables 
        X_list = list(processed_df.columns)
        X = processed_df[X_list]
        X = sm.add_constant(X)  # Adds an intercept term
        
        # Store both final X_list (order preserved) and final y variables in global lists
        FINAL_HEART_REGRESSORS = X_list
        FINAL_HEART_Y = ['Heart Attack Risk']
        
        # Check if final preprocessing has been done correctly
        save_dataframe(processed_df, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "heart_preprocessed_final.csv")
        
        return processed_df
    
class oos_testing:
    def split_test_train_data(self, df: pd.DataFrame, target_column, test_size=0.2, random_state=42):
        """
        Perform out-of-sample testing by dividing the dataset into training and testing data.
        """
        # Separate the target variable (y) and the features (X)
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Make PyTorch Tensors from numpy arrays or pandas DataFrames
        X_train_tensor = torch.tensor(X_train.values.astype(np.float32)) if isinstance(X_train, pd.DataFrame) else torch.tensor(X_train.astype(np.float32))
        y_train_tensor = torch.tensor(y_train.values.astype(np.int64)) if isinstance(y_train, pd.Series) else torch.tensor(y_train.astype(np.int64))
        X_test_tensor = torch.tensor(X_test.values.astype(np.float32)) if isinstance(X_test, pd.DataFrame) else torch.tensor(X_test.astype(np.float32))
        y_test_tensor = torch.tensor(y_test.values.astype(np.int64)) if isinstance(y_test, pd.Series) else torch.tensor(y_test.astype(np.int64))

        # Define datasets and DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        print("Training dataset shape:", X_train.shape, y_train.shape)
        print("Validation dataset shape:", X_test.shape, y_test.shape)
        print("Training dataset type:", type(X_train), type(y_train))
        print("Validation dataset type:", type(X_test), type(y_test))
        print('')
        print("Object type of current x and y variables:", type(X_train))
        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)

        return X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader, val_loader