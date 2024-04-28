import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, log_loss

from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from typing import List

from .raw_dataframe_preprocessor import return_final_variables

class ColumnRearranger:
    def bootstrap_to_match(self, df_left: pd.DataFrame, df_right: pd.DataFrame):
        """
        Adjusts the number of rows in df_right to match the number of rows in df_left
        by random sampling with replacement (bootstrapping).

        Args:
        df_left (pd.DataFrame): DataFrame with fewer or equal rows.
        df_right (pd.DataFrame): DataFrame from which rows will be sampled.

        Returns:
        pd.DataFrame: A new DataFrame sampled from df_right with the same number of rows as df_left.
        """
        # Check if df_right already has the same number of rows
        if len(df_left) == len(df_right):
            return df_right

        # Randomly sample rows from df_right with replacement
        df_right_bootstrapped = df_right.sample(n=len(df_left), replace=True, random_state=42)

        return df_right_bootstrapped
    
    def return_optimal_rearrangement(self, df_left, df_right) -> pd.DataFrame:
        """
        This function rearranges the columns of df_right to best align with the column arrangement of df_left.
        It uses linear regression for numeric predictors and logistic regression for categorical predictors.
        It finally applies the Hungarian algorithm to rearrange the right table's columns such that the total correlation between i=j columns is maximized.
        """
        X_left, X_right, y_left, y_right = return_final_variables()  # Assuming this function exists and returns appropriate lists
        n, m = len(X_left), len(X_right)
        print(f"n={n}, m={m}")

        # check for row number equality
        if len(df_left) != len(df_right):
            raise ValueError("Equate the row numbers first.")

        # Initialize cost matrix with extra padding if necessary
        max_dim = max(n, m)
        cost_matrix = np.full((max_dim, max_dim), float('inf'))  # Initialize with high cost for padding
        
        # Identify categorical columns
        categorical_columns = df_left.select_dtypes(include=['object']).columns.tolist()

        # Fill the cost matrix
        for i, col_left in enumerate(X_left):
            for j, col_right in enumerate(X_right):
                y = df_left[col_left].values.ravel()  # Flatten y to 1D
                X = df_right[col_right].values.reshape(-1, 1)  # Reshape X to 2D

                if col_left in categorical_columns and len(np.unique(y)) > 1:
                    # Use logistic regression for categorical outcomes, ensuring y has more than one unique value
                    model = LogisticRegression(solver='lbfgs', max_iter=1000)
                    model.fit(X, y)
                    prediction = model.predict_proba(X)[:, 1]  # Get probability for the positive class
                elif len(np.unique(y)) > 1:
                    # Use linear regression for non-categorical types of data, ensuring y has more than one unique value
                    model = LinearRegression()
                    model.fit(X, y)
                    prediction = model.predict(X)
                else:
                    # If y is constant, no need to fit a model
                    prediction = np.full_like(y, fill_value=np.mean(y))

                # Check if prediction is constant
                if np.std(prediction) == 0 or np.std(y) == 0:
                    corr = 0  # Set correlation to 0 if prediction or y is constant
                else:
                    # Calculate correlation and handle any potential nan values
                    corr, _ = pearsonr(prediction, y)
                    if np.isnan(corr):
                        corr = 0

                # Fill the cost matrix, using only valid indices
                if i < n and j < m:
                    cost_matrix[i, j] = -corr  # Negative correlation to minimize the cost
                
        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix[:n, :m])  # Limit to the real dimensions

        # Rearrange columns according to the optimal assignment
        rearranged_columns = [X_right[j] for j in col_ind]
        remaining_columns = [col for col in X_right if col not in rearranged_columns]
        rearranged_df_right = df_right[rearranged_columns + remaining_columns].copy()

        return rearranged_df_right
    
    def compute_average_correlation(self, df_left, df_right):
        """
        Computes the average correlation between each corresponding column pair (i=j) of df_left and df_right.
        It utilizes a covariance matrix to account for both linear and logistic correlations.
        """
        # check for row number equality
        if len(df_left) != len(df_right):
            raise ValueError("Equate the row numbers first.")
        
        # Identify categorical columns
        categorical_columns = df_left.select_dtypes(include=['object']).columns.tolist()

        correlations = []
        # Calculate correlations for each column
        for col_left, col_right in zip(df_left.columns, df_right.columns):
            y = df_left[col_left].values.ravel()  # Flatten y to 1D
            X = df_right[col_right].values.reshape(-1, 1)  # Reshape X to 2D
            
            if col_left in categorical_columns and len(np.unique(y)) > 1:
                # Use logistic regression for categorical columns
                model = LogisticRegression(solver='lbfgs', max_iter=1000)
                model.fit(X, y)
                prediction_prob = model.predict_proba(X)[:, 1]  # Probability for the positive class
                score = log_loss(y, prediction_prob)
                # Convert log loss to a correlation-like metric for comparison purposes
                correlation = 1 - score / log_loss(y, np.full_like(y, fill_value=y.mean()))
            elif len(np.unique(y)) > 1:
                # Use linear regression for numerical columns
                model = LinearRegression()
                model.fit(X, y)
                prediction = model.predict(X)
                # Calculate correlation and handle any potential nan values
                correlation, _ = pearsonr(prediction, y)
                if np.isnan(correlation):
                    correlation = 0
            else:
                # If y is constant, set correlation to 0
                correlation = 0
            
            correlations.append(correlation)
        
        # Compute the average of the correlations
        average_correlation = np.mean(correlations)
        return average_correlation
    
    def visualize_comparison(self, avg_corr_before, avg_corr_after):
        """
        Generates a bar chart to compare the average correlations before and after rearrangement.
        
        Args:
        avg_corr_before (float): The average correlation before rearrangement.
        avg_corr_after (float): The average correlation after rearrangement.
        """
        # Define labels, positions, and the values for the bars
        labels = ['Before Rearrangement', 'After Rearrangement']
        positions = range(len(labels))
        values = [avg_corr_before, avg_corr_after]
        
        # Create the bar chart
        plt.figure(figsize=(10, 5))
        plt.bar(positions, values, color=['blue', 'green'])
        plt.xlabel('Condition')
        plt.ylabel('Average Correlation')
        plt.title('Comparison of Average Correlation Before and After Rearrangement')
        plt.xticks(positions, labels)
        plt.ylim(0, 1)  # Assuming correlation values are between 0 and 1
        
        # Add the actual values on top of the bars for clarity
        for pos, value in zip(positions, values):
            plt.text(pos, value + 0.01, f"{value:.2f}", ha='center', va='bottom')
        
        # Display the plot
        plt.show(block=False)
        
        # Save the plot to a file
        plt.savefig('./PVM/Plots/column_rearrangement_test.png')
        plt.close()