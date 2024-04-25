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

# from .VQVAE import Trainer, DataFrameDataset

class RowMatcher:
    def create_index(self, model, dataframe, device='gpu'):
        # Fetch embeddings using the model
        model.eval()
        embeddings = []
        with torch.no_grad():
            for index in dataframe['encoded_vector'].unique():
                # Fetch embedding corresponding to each unique index
                embedding = model.vq_vae.embedding(torch.tensor([index], device=device)).squeeze().gpu().numpy()
                embeddings.append(embedding)
        
        embeddings = np.stack(embeddings)

        # Initialize the HNSW index
        dim = embeddings.shape[1]
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        index.add_items(embeddings)

        return index
    
    def retrieve_similar(df_randhie, df_heart, randhie_model, heart_model, heart_index, device='cpu'):
        # Generate embeddings for the randhie data
        randhie_embeddings = []
        randhie_model.eval()
        with torch.no_grad():
            for idx in df_randhie['encoded_vector']:
                embedding = randhie_model.vq_vae.embedding(torch.tensor([idx], device=device)).squeeze().cpu().numpy()
                randhie_embeddings.append(embedding)
        
        randhie_embeddings = np.array(randhie_embeddings)

        # Perform HNSW retrieval
        labels, distances = heart_index.knn_query(randhie_embeddings, k=1)  # Retrieve the top 1 nearest neighbor

        # Extract matching rows from the heart dataframe
        similar_rows = df_heart.iloc[labels.flatten()]

        # Combine the data for easier inspection or further processing
        combined_data = df_randhie.copy()
        combined_data['matched_index'] = labels.flatten()
        combined_data['distance'] = distances.flatten()
        combined_data['matched_row'] = similar_rows.values.tolist()

        return combined_data