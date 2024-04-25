import os
import pandas as pd
import numpy as np

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from typing import List

from .raw_dataframe_preprocessor import return_final_variables

class ColumnRearranger:
    def return_optimal_rearrangement(self, df_left, df_right):
        """
        This function rearranges the columns of df_right to best align with the column arrangement of df_left.
        It uses linear regression for numeric predictors and logistic regression for categorical predictors.
        It finally applies the Hungarian algorithm to rearrange the right table's columns such that the total correlation between i=j columns is maximized.
        """
        X_left, X_right, y_left, y_right = return_final_variables()
        n, m = len(X_left), len(X_right)
        print(f"n={n}\n m={m}")
        if n != m:
            raise ValueError("The number of columns specified for left and right DataFrames must be equal.")

        # Initialize cost matrix
        cost_matrix = np.zeros((n, m))
        label_encoders = {col: LabelEncoder().fit(df_left[col]) for col in X_left if df_left[col].dtype == object}

        # Fill the cost matrix
        for i, col_left in enumerate(X_left):
            for j, col_right in enumerate(X_right):
                if df_left[col_left].dtype == object or len(df_left[col_left].unique()) > 2:
                    y = label_encoders[col_left].transform(df_left[col_left]) if df_left[col_left].dtype == object else df_left[col_left]
                    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                else:
                    y = df_left[col_left]
                    model = LinearRegression()

                model.fit(df_right[[col_right]], y)
                prediction = model.predict(df_right[[col_right]])
                corr = model.score(df_right[[col_right]], y) if isinstance(model, LogisticRegression) else pearsonr(prediction.flatten(), y)[0]
                cost_matrix[i, j] = -corr  # Negative for maximization

        # Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        rearranged_columns = [X_right[idx] for idx in col_ind]

        # Rearrange columns according to the optimal assignment
        rearranged_df_right = df_right[rearranged_columns].copy()

        return rearranged_df_right