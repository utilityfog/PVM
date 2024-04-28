import hnswlib
import pandas as pd
import numpy as np
import torch
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from scipy.spatial.distance import cdist

class RowMatcher:
    def __init__(self):
        self.used_indices = set()

    def retrieve_similar(self, df_randhie, df_heart):
        """Retrieve most similar rows based on encoded vectors."""
        combined_data = pd.DataFrame()

        randhie_vectors = np.vstack(df_randhie['encoded_vector'].apply(np.array))
        heart_vectors = np.vstack(df_heart['encoded_vector'].apply(np.array))

        distances = cdist(randhie_vectors, heart_vectors, metric='euclidean')

        for i, distance_vector in enumerate(distances):
            min_index = np.argmin([dist if idx not in self.used_indices else np.inf for idx, dist in enumerate(distance_vector)])
            distance = distance_vector[min_index]

            # Check and append only if index has not been used
            if min_index not in self.used_indices:
                self.used_indices.add(min_index)
                similar_row = df_heart.iloc[min_index]
                combined_row = pd.concat([df_randhie.iloc[i], similar_row], axis=0)
                combined_row['matched_index'] = min_index
                combined_row['distance'] = distance
                combined_data = combined_data._append(combined_row, ignore_index=True)

        return combined_data