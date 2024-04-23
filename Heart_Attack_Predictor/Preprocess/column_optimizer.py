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

from .raw_dataframe_preprocessor import FINAL_RANDHIE_REGRESSORS

class ColumnRearranger:
    def return_optimal_rearrangement(self, df_left, df_right, X_left=FINAL_RANDHIE_REGRESSORS, X_right=[]) -> pd.DataFrame:
        """
        function that receives both the left and right dataframes as well as a list of regressors for each, rearranges the columns of the right dataframe
        such that it best aligns with the column arrangement of the left dataframe and then returns the rearranged right dataframe.
        """
        # This function is necessary because vectorization of rows depends on column arrangement because each column represents a dimension
        n, m = len(X_left), len(X_right)  # Number of columns specified in each dataset
        if n > m:
            raise ValueError("There are more regressors specified in the left DataFrame than in the right DataFrame.")
        
        # Initialize cost matrix
        cost_matrix = np.zeros((n, m))
        label_encoders = {col: LabelEncoder().fit(df_left[col]) for col in X_left if df_left[col].dtype == object}

        # Calculate correlation coefficients to fill the cost matrix
        for i, col_left in enumerate(X_left):
            for j, col_right in enumerate(X_right):
                # Determine if the target column is categorical with more than two categories
                if df_left[col_left].dtype == object or len(df_left[col_left].unique()) > 2:
                    # Assume categorical data is properly encoded or encode it as needed
                    if df_left[col_left].dtype == object:
                        y = label_encoders[col_left].transform(df_left[col_left])
                    else:
                        y = df_left[col_left]

                    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                else:
                    model = LinearRegression()

                model.fit(df_right[[col_right]], y)
                prediction = model.predict(df_right[[col_right]])

                if isinstance(model, LogisticRegression):
                    # Use R^2 as the correlation measure for categorical outcomes
                    score = model.score(df_right[[col_right]], y)
                    corr = 2 * score - 1  # Rescale R^2 to range from -1 to 1
                else:
                    corr, _ = pearsonr(prediction.flatten(), y)

                cost_matrix[i, j] = -corr  # Use negative correlation because we minimize

        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Rearrange columns according to the optimal assignment
        rearranged_columns = [X_right[idx] for idx in col_ind]
        rearranged_df_right = df_right[rearranged_columns].copy()

        return rearranged_df_right