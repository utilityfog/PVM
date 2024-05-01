import os
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from torch.utils.data import DataLoader, TensorDataset

from .globals import RANDHIE_CATEGORICAL_VARIABLES, RANDHIE_NUMERIC_VARIABLES, HEART_CATEGORICAL_VARIABLES, HEART_NUMERIC_VARIABLES

FINAL_RANDHIE_REGRESSORS = []
FINAL_RANDHIE_Y = []

FINAL_HEART_REGRESSORS = []
FINAL_HEART_Y = []

FINAL_RANDHIE_PREDICTOR_DATAFRAME = None
FINAL_HEART_PREDICTOR_DATAFRAME = None

def return_final_variables():
    return FINAL_RANDHIE_REGRESSORS, FINAL_HEART_REGRESSORS, FINAL_RANDHIE_Y, FINAL_HEART_Y

def return_final_predictor_dataframes():
    return FINAL_RANDHIE_PREDICTOR_DATAFRAME, FINAL_HEART_PREDICTOR_DATAFRAME

def update_heart_final_predictors(df, columns):
    global FINAL_HEART_PREDICTOR_DATAFRAME
    FINAL_HEART_PREDICTOR_DATAFRAME = df
    global FINAL_HEART_REGRESSORS
    FINAL_HEART_REGRESSORS = columns

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
    
def plot_and_save_covariance_matrix(df, filepath):
    """
    Computes the covariance matrix of the columns in the given DataFrame, plots a heatmap of it,
    and saves the plot as an image file.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numeric data.
    - filename (str): The name of the file to save the image.

    Returns:
    - None: The function saves the heatmap image of the covariance matrix.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
    # Calculate the covariance matrix
    covariance_matrix = df.cov()
    
    # Create a heatmap of the covariance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Covariance Matrix Heatmap')
    
    # Save the plot to a file
    plt.savefig(filepath)
    plt.close()

    print(f"Heatmap saved as {filepath}")
    
def save_OLS_to_plot(regression_results, is_initial):
    # Plotting regression results
    for target, results in regression_results.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        print(f"results: {results.summary()}")
        coefs = results.params
        conf = results.conf_int()
        err = (conf[1] - conf[0]) / 2

        ax.errorbar(coefs.index, coefs, yerr=err, fmt='o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0)
        ax.set_ylabel('Coefficient')
        if is_initial:
            ax.set_title(f'Initial OLS Regression Results for {target}')
        else:
            ax.set_title(f'Augmented OLS Regression Results for {target}')
        
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_xticks(range(len(coefs)))  # Ensure x-ticks are correctly set
        ax.set_xticklabels(coefs.index, rotation=90)

        # Annotating R-squared and F-statistic
        stats_text = f'R-squared: {results.rsquared:.3f}\nF-statistic: {results.fvalue:.3f}\nProb (F-statistic): {results.f_pvalue:.4f}'
        ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', verticalalignment='top', fontsize=12)

        plt.tight_layout()
        
        if is_initial:
            plt.savefig(f'./PVM/Plots/regression_results_{target}_initial.png')
        else:
            plt.savefig(f'./PVM/Plots/regression_results_{target}_augmented.png')
            
        plt.close()

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
        X = sm.add_constant(X)  # Adds a constant intercept

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
        # save_dataframe(df_cleaned, os.getcwd()+"/PVM/Datasets", "randhie_preprocessed0.csv")
        
        # Average the numeric column values at the patient (zper) level since we are not interested in time and the RANDHIE experiment has 5 individual year observations for each patient
        collapsed_df = self.average_by_unique_patient(df_cleaned, "zper", RANDHIE_CATEGORICAL_VARIABLES)
        print(f"AVERAGED: {collapsed_df.head()}")
        # Test 1
        # save_dataframe(collapsed_df, os.getcwd()+"/PVM/Datasets", "randhie_preprocessed1.csv")
        
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
        # save_dataframe(collapsed_df, os.getcwd()+"/PVM/Datasets", "randhie_preprocessed2.csv")
        
        # Standardize Numeric Columns: standardized_df = standardize_df(avg_df)
        new_numeric_columns = RANDHIE_NUMERIC_VARIABLES
        new_categorical_columns = RANDHIE_CATEGORICAL_VARIABLES
        standardized_df = standardize_dataframe(collapsed_df, new_numeric_columns, new_categorical_columns)
        print(f"STANDARDIZED - NEW: {standardized_df.head()}")
        # Add xage, linc, log_med_exp, log_inpatient_exp without standardization
        standardized_df['linc'] = collapsed_df['linc']
        standardized_df['xage'] = collapsed_df['xage']
        standardized_df['log_outpatient_exp'] = collapsed_df['log_med_exp']
        standardized_df['log_inpatient_exp'] = collapsed_df['log_inpatient_exp']
        # Test 3
        # save_dataframe(standardized_df, os.getcwd()+"/PVM/Datasets", "randhie_preprocessed3.csv")
        
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
        
        # Drop unnecessary columns
        standardized_df.drop(['child', 'fchild'], axis=1, inplace=True)
        for item in ['child', 'fchild']:
            new_categorical_columns.remove(item)
        # Test 4
        # save_dataframe(standardized_df, os.getcwd()+"/PVM/Datasets", "randhie_preprocessed4.csv")
        
        # One Hot Encoding categorical variables
        encoded_df = encode_categorical(standardized_df, new_categorical_columns)
        print(f"ONE HOT ENCODED (CATEGORICAL VARS): {encoded_df.head()}")
        
        # Replacing the randhie df's categorical variables with one hot encoded variables
        processed_df = replace_encoded_categorical(standardized_df, encoded_df, new_categorical_columns)
        print(f"PROCESSED: {processed_df.head()}")

        # Define independent variables based on the paper's model and available data: 
            # Excluding variables that are known to be endogenous (e.g. if someone makes a lot of hospital visits, obviously it will have a positive causal relationship with their quantity demanded for medical care even if there are confounding variables that may affect the number of times they visit the hospital)
            # Excluding variables that are a deterministic function of another to prevent perfect multicolinearity; no inclusion of both linc and income
            # We use one hot encoding instead of dummy encoding because vectorization is affected by that choice. We want to capture maximum information for each vectorized row
        processed_df.drop(['income', 'year', 'outpdol', 'drugdol', 'suppdol', 'mentdol', 'inpdol', 'meddol', 'totadm', 'num', 'ghindx', 'logc', 'fmde', 'lnmeddol', 'binexp', 'zper'], axis=1, inplace=True)
        print(f"processed_df columns: {list(processed_df.columns)}")
        # 'is_positive_med_exp', 'is_positive_inpatient_exp', 'is_only_outpatient_exp'
        y_list = ['log_outpatient_exp', 'log_inpatient_exp']
        X = processed_df.drop(y_list+['log_med_exp', 'is_positive_med_exp', 'is_positive_inpatient_exp', 'is_only_outpatient_exp'], axis=1)
        print(f"final randhie X: {X.head()}")
        
        # Final RANDHIE predictors
        X_list = list(X.columns)
        print(f"length of final randhie X: {len(X_list)}")
        # ['coins', 'person_type_adult', 'person_type_fchild', 'person_type_mchild', 'hlthg_0', 'hlthg_1', 'hlthf_0', 'hlthf_1', 'hlthp_0', 'hlthp_1', 'female_0', 'female_1', 'site_2', 'site_3', 'site_4', 'site_5', 'site_6', 'tookphys_0', 'tookphys_1', 'plan', 'xage', 'educdec', 'time', 'disea', 'physlm', 'mdeoff', 'lfam', 'lpi', 'logc', 'xghindx', 'linc', 'lnum', 'black', 'mhi']
        X = sm.add_constant(X)  # Adds an intercept term
        
        # DataFrame for final predictors of randhie dataset
        # save_dataframe(X, os.getcwd()+"/PVM/Datasets", "randhie_preprocessed_X.csv")
        # plot_and_save_covariance_matrix(X, filepath='./PVM/Plots/randhie_covariance_matrix.png')
        
        # Check if final preprocessing has been done correctly; includes both X and y
        # save_dataframe(processed_df, os.getcwd()+"/PVM/Datasets", "randhie_preprocessed_final.csv")
        
        regression_results = {}
        # FOUR EQUATION MODEL ACCORDING TO PAPER: Health Insurance and the Demand for Medical Care
        
        ### REMOVED due to all instances of is_positive_med_exp only being 1 after being preprocessed; this is anticipated as we are have summarized 5 years of data into 1
        # Equation 1: Lasso model for zero versus positive medical expenses
        # lasso_log_model_1 = LogisticRegression(penalty='l1', solver='liblinear')
        # lasso_log_model_1.fit(X, processed_df['is_positive_med_exp'])
        # regression_results['is_positive_med_exp'] = lasso_log_model_1
        
        # Equation 2: Lasso model for having zero versus positive inpatient expense, given positive use of medical services
        df_pos_med_exp = processed_df[processed_df['is_positive_med_exp'] == 1]  # Filter for positive medical use
        lasso_log_model_2 = LogisticRegression(penalty='l1', solver='liblinear')
        lasso_log_model_2.fit(X, df_pos_med_exp['is_positive_inpatient_exp'])
        # regression_results['is_positive_inpatient_exp'] = lasso_log_model_2
        
        # Equation 3: OLS regression for log of positive medical expenses if only outpatient services are used
        df_only_outpatient_exp = processed_df[processed_df['is_only_outpatient_exp'] == 1]
        model_3 = OLS(df_only_outpatient_exp['log_med_exp'], X.loc[df_only_outpatient_exp.index]).fit()
        regression_results['log_outpatient_exp'] = model_3
        
        # Equation 4: OLS regression for log of medical expenses for those with any inpatient expenses
        df_pos_inpatient_exp = processed_df[processed_df['is_positive_inpatient_exp'] == 1]
        model_4 = OLS(df_pos_inpatient_exp['log_inpatient_exp'], X.loc[df_pos_inpatient_exp.index]).fit()
        regression_results['log_inpatient_exp'] = model_4
        
        # Print summaries of the models
        
        # Equation 1
        # print("\nEquation 1: Lasso model for zero versus positive medical expenses")
        # print("Lasso Logistic Regression Model Coefficients:")
        # print(lasso_log_model_1.coef_)
        # print("Intercept:", lasso_log_model_1.intercept_)
        # # Generating predictions to use in classification report
        # predictions = lasso_log_model_1.predict(X)
        # # Classification report
        # print("\nClassification Report:")
        # print(classification_report(df_pos_med_exp['is_positive_med_exp'], predictions))
        # # Confusion Matrix
        # print("\nConfusion Matrix:")
        # print(confusion_matrix(df_pos_med_exp['is_positive_med_exp'], predictions))
        
        # Equation 2
        print("\nEquation 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services")
        print("Lasso Logistic Regression Model Coefficients:")
        print(lasso_log_model_2.coef_)
        print("Intercept:", lasso_log_model_2.intercept_)
        # Generating predictions to use in classification report
        predictions = lasso_log_model_2.predict(X)
        # Classification report
        print("\nClassification Report:")
        print(classification_report(df_pos_med_exp['is_positive_inpatient_exp'], predictions))
        # Confusion Matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(df_pos_med_exp['is_positive_inpatient_exp'], predictions))
        
        # Equation 3
        print("\nEquation 3: OLS regression for log of positive medical expenses if only outpatient services are used")
        print(model_3.summary())
        
        # Equation 4
        print("\nEquation 4: OLS regression for log of medical expenses for those with any inpatient expenses")
        print(model_4.summary())
        
        # Save as plot
        save_OLS_to_plot(regression_results, True)
        
        # Drop the constant intercept before returning or saving X
        X.drop('const', axis=1, inplace=True)
        
        # Store both final X_list (order static) and final y variables in global lists
        global FINAL_RANDHIE_REGRESSORS 
        FINAL_RANDHIE_REGRESSORS = X_list
        global FINAL_RANDHIE_Y
        FINAL_RANDHIE_Y = y_list
        
        # Store to global df
        global FINAL_RANDHIE_PREDICTOR_DATAFRAME
        FINAL_RANDHIE_PREDICTOR_DATAFRAME = X
        
        return processed_df, X
    
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
    def preprocess(self, data_input):
        """
        Method that pre-processes the heart_attack_prediction_dataset.csv dataset
        
        CleaningDecisions:
        1. Seperate Systolic and Diastolic blood pressure into seperate columns 
        2. Create an interaction term between systolic and diastolic.
        3. Drop 'Sex_Male', 'Diet_Average', 'Country_Argentina' to prevent multicolinearity.
        4. Set categorical variables to category type.
        
        Returns: pd.DataFrame
        """
        save_covariance_matrix = True
        # Read
        if isinstance(data_input, str):
            df = pd.read_csv(data_input)
        elif isinstance(data_input, pd.DataFrame):
            df = data_input
            save_covariance_matrix = False
        else:
            raise ValueError("data_input must be a filepath string or a pandas DataFrame")
        print(f"Raw DataFrame: {df.head()}")
        
        # Remove rows with NaN or inf values
        df_cleaned = remove_nan_or_inf(df)
        print(f"DataFrame after removing NaN and inf values: {df_cleaned.head()}")
        # Test 0 
        # save_dataframe(df_cleaned, os.getcwd()+"/PVM/Datasets", "heart_preprocessed0.csv")
        # Get Log Income
        df_cleaned['Log_Income'] = np.log(df_cleaned['Income'])
        
        ### Seperate systolic and diastolic blood pressure into their own variables and create an interaction term.
        # Split the 'Blood Pressure' column into 'Systolic' and 'Diastolic'
        df_cleaned[['Systolic', 'Diastolic']] = df_cleaned['Blood Pressure'].str.split('/', expand=True).astype(int)
        
        # Create interaction term by multiplying Systolic and Diastolic pressures
        # df_cleaned['BP_Interaction'] = df_cleaned['Systolic'] * df_cleaned['Diastolic']
        
        # Drop unnecessary variables
        df_cleaned.drop(['Patient ID', 'Blood Pressure', 'Income', 'Continent', 'Hemisphere'], axis=1, inplace=True)
        
        # Standardize Numeric Columns: standardized_df = standardize_df(avg_df)
        numeric_columns = HEART_NUMERIC_VARIABLES + ['Systolic', 'Diastolic']
        categorical_columns = HEART_CATEGORICAL_VARIABLES
        standardized_df = standardize_dataframe(df_cleaned, numeric_columns, categorical_columns)
        print(f"STANDARDIZED - NEW: {standardized_df.head()}")
        standardized_df['Age'] = df_cleaned['Age'] # Add age without standardization to engineer the contribution of Age the VAE Vector Encoding process since age is a common column
        
        # Test 1
        # save_dataframe(standardized_df, os.getcwd()+"/PVM/Datasets", "heart_preprocessed1.csv")
        
        # One Hot Encoding categorical variables
        encoded_df = encode_categorical(standardized_df, categorical_columns)
        print(f"ONE HOT ENCODED (CATEGORICAL VARS): {encoded_df.head()}")
        
        # Replacing the heart df's categorical variables with one hot encoded variables
        processed_df = replace_encoded_categorical(standardized_df, encoded_df, categorical_columns)
        print(f"PROCESSED: {processed_df.head()}")
        
        global FINAL_HEART_Y
        FINAL_HEART_Y = ['Heart Attack Risk']
        
        y = processed_df[FINAL_HEART_Y]
        
        X = processed_df.drop(FINAL_HEART_Y, axis=1)

        # Define independent variables
        X_list = list(X.columns)
        print(f"final heart X: {X.head()}")
        print(f"length of final heart X: {len(X.columns)}")
        global FINAL_HEART_REGRESSORS
        FINAL_HEART_REGRESSORS = X_list
        
        # Tensorize
        X_tensor = torch.tensor(X.values.astype(np.float32)) if isinstance(X, pd.DataFrame) else torch.tensor(X.astype(np.float32))
        y_tensor = torch.tensor(y.values.astype(np.float32)) if isinstance(y, pd.DataFrame) else torch.tensor(y.astype(np.float32))
        
        # Cov Matrix for final predictors of heart dataset
        if save_covariance_matrix:
            plot_and_save_covariance_matrix(X, filepath='./PVM/Plots/heart_covariance_matrix.png')
        
        return processed_df, X, y, X_tensor, y_tensor

class OOS(KFold):
    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Perform out-of-sample testing by dividing the dataset into training and testing data.
        """
        # Split the data into training and testing sets
        fold_indexes = super().split(X)
        
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        for train_index, test_index in fold_indexes:
            df_train_value, df_test_value = X.iloc[train_index], X.iloc[test_index]
            df_train = df_train._append(df_train_value, ignore_index=True)
            df_test = df_test._append(df_test_value, ignore_index=True)
        
        df_train_tensor_raw = None
        df_test_tensor_raw = None
        
        try:
            # Make PyTorch Tensors from numpy arrays or pandas DataFrames
            df_train_tensor_raw = torch.tensor(df_train.values.astype(np.float32)) if isinstance(df_train, pd.DataFrame) else torch.tensor(df_train.astype(np.float32))
            df_test_tensor_raw = torch.tensor(df_test.values.astype(np.float32)) if isinstance(df_test, pd.DataFrame) else torch.tensor(df_test.astype(np.float32))
        except:
            print("Not yet ready for tensorization!")
        
        return super().split(X)