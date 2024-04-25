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

from .VQVAE import Trainer, DataFrameDataset

from Preprocess.raw_dataframe_preprocessor import FINAL_RANDHIE_REGRESSORS, FINAL_RANDHIE_Y

class DataFrameEncoder:
    def __init__(self):
        # Train and assign the randhie and heart VQ-VAE models
        self.trainer = Trainer()
        self.device = None
        self.randhie_model = None
        self.heart_model = None

    def train_and_assign_models(self):
        # Train and assign the randhie and heart VQ-VAE models
        self.device, self.randhie_model, self.heart_model = self.trainer.train()

    def encode_dataframe(self, df, model):
        """Encodes all rows in a dataframe using a trained VQ-VAE model."""
        dataset = DataFrameDataset(df)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        encoded_vectors = []

        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                _, _, _, encoding_indices = model(data)
                encoded_vectors.append(encoding_indices.squeeze().cpu().numpy())

        # Return encoded vectors as a new DataFrame column
        return df.assign(encoded_vector=encoded_vectors)

    def save_model(self, model, path):
        # Save the model to the specified path
        torch.save(model.state_dict(), path)

    def load_and_encode_dataframes(self, randhie_df, heart_df):
        # Load the trained models (this assumes the models have been saved after training)
        self.randhie_model.load_state_dict(torch.load('randhie_model.pth'))
        self.heart_model.load_state_dict(torch.load('heart_model.pth'))

        # Encode the dataframes
        encoded_randhie_df = self.encode_dataframe(randhie_df, self.randhie_model)
        encoded_heart_df = self.encode_dataframe(heart_df, self.heart_model)

        return encoded_randhie_df, encoded_heart_df