# Future imports must come first
from __future__ import print_function

# Standard library imports
from six.moves import xrange

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import statsmodels.api as sm
import torch
import torch.nn.functional as F
import torchvision
import umap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import Probit
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, utils

from Preprocess import raw_dataframe_preprocessor

# Custom adjustments and functions (e.g., using savgol_filter directly from scipy.signal)
savgol_filter = sg.savgol_filter
make_grid = utils.make_grid

### Re-Implementation of Aäron van den Oord's VQ VAE to suit tabular data instead of image data

# HYPERPARAMETERS
BATCH_SIZE = 256
NUM_TRAINING_UPDATES = 15000

NUM_HIDDENS = 128
NUM_RESIDUAL_HIDDENS = 32
NUM_RESIDUAL_LAYERS = 2

EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512

COMMITMENT_COST = 0.25

DECAY = 0.99

LEARNING_RATE = 1e-3

class DataFramePreprocessor:
    def load_and_split(self):
        """
        This method must be called after the initial dataframe preprocessing stage!
        """
        ### RANDHIE
        randhie_columns, heart_columns = raw_dataframe_preprocessor.return_final_predictor_dataframes()
        randhie_train, randhie_validation = train_test_split(randhie_columns, test_size=0.25, random_state=42)
        
        # Converting the split dataframes into datasets
        randhie_training_data = DataFrameDataset(randhie_train)
        randhie_validation_data = DataFrameDataset(randhie_validation)
        
        # Bootstrap due to lack of data: Create weights for each row for the WeightedRandomSampler
        randhie_weights = np.ones(len(randhie_training_data))
        
        # Create a sampler instance
        randhie_sampler = WeightedRandomSampler(randhie_weights, num_samples=len(randhie_weights)*5, replacement=True)

        # Create DataLoader instances
        randhie_training_loader = DataLoader(randhie_training_data, 
                                    batch_size=BATCH_SIZE, 
                                    sampler=randhie_sampler,
                                    pin_memory=True)

        randhie_validation_loader = DataLoader(randhie_validation_data,
                                    batch_size=32,
                                    shuffle=True,
                                    pin_memory=True)
        
        # Compute the variance of the numerical data in the heart training dataset
        randhie_data_numeric = randhie_training_data.dataframe.select_dtypes(include=[np.number])
        # randhie_data_variance = np.var(randhie_data_numeric, axis=0)
        # Flatten the numerical data and compute the overall variance
        randhie_data_flattened = randhie_data_numeric.values.flatten()
        randhie_total_variance = np.var(randhie_data_flattened)
        
        ### HEART
        heart_train, heart_validation = train_test_split(heart_columns, test_size=0.25, random_state=42)
        
        # Converting the split dataframes into datasets
        heart_training_data = DataFrameDataset(heart_train)
        heart_validation_data = DataFrameDataset(heart_validation)
        
        # Bootstrap due to lack of data: Create weights for each row for the WeightedRandomSampler
        heart_weights = np.ones(len(heart_training_data))
        
        # Create a sampler instance
        heart_sampler = WeightedRandomSampler(heart_weights, num_samples=len(heart_weights)*5, replacement=True)

        # Create DataLoader instances
        heart_training_loader = DataLoader(heart_training_data, 
                                    batch_size=BATCH_SIZE, 
                                    sampler=heart_sampler,
                                    pin_memory=True)

        heart_validation_loader = DataLoader(heart_validation_data,
                                    batch_size=32,
                                    shuffle=True,
                                    pin_memory=True)
        
        # Compute the variance of the numerical data in the heart training dataset
        heart_data_numeric = heart_training_data.dataframe.select_dtypes(include=[np.number])
        # heart_data_variance = np.var(heart_data_numeric, axis=0)
        # Flatten the numerical data and compute the overall variance
        heart_data_flattened = heart_data_numeric.values.flatten()
        heart_total_variance = np.var(heart_data_flattened)
        
        return randhie_training_loader, randhie_validation_loader, heart_training_loader, heart_validation_loader, randhie_total_variance, heart_total_variance

class DataFrameDataset(Dataset):
    """Custom Dataset for loading rows from a pandas DataFrame."""
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx].values.astype('float32')
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize the embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # Reshape input if necessary
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances between input features and embeddings
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find the nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize by replacing the input with its nearest embedding
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity to monitor usage of embeddings
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encoding_indices
    
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(in_channels, num_residual_hiddens),
            nn.ReLU(True),
            nn.Linear(num_residual_hiddens, num_hiddens)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, num_features, num_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(num_features, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, embedding_dim)  # Adjust to match embedding_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Redefine the model's Decoder as well to match the structure
class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_features):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)  # First layer to handle embedding_dim
        self.fc2 = nn.Linear(embedding_dim, num_features)  # Output matches the original num_features

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Model(nn.Module):
    def __init__(self, num_features, num_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super(Model, self).__init__()
        self.encoder = Encoder(num_features, num_hiddens, embedding_dim)
        self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_features)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity
    
class Trainer:
    def train(self):
        """
        Pretrain VQ VAE
        """
        # Helper function to convert tensors to dataframe
        def tensor_to_df(tensor, columns):
            return pd.DataFrame(tensor.detach().cpu().numpy(), columns=columns)
        
        # Load data loaders for both randhie and heart
        dataframe_preprocessor = DataFramePreprocessor()
        randhie_columns, heart_columns, _, _ = raw_dataframe_preprocessor.return_final_variables()
        randhie_training_loader, randhie_validation_loader, heart_training_loader, heart_validation_loader, randhie_variance, heart_variance = dataframe_preprocessor.load_and_split()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ### Train VQ VAE for randhie
        randhie_model = Model(39, NUM_HIDDENS,
                NUM_EMBEDDINGS, EMBEDDING_DIM, 
                COMMITMENT_COST).to(device)
        # Optimizer
        randhie_optimizer = optim.Adam(randhie_model.parameters(), lr=LEARNING_RATE, amsgrad=False)
        
        randhie_model.train()
        randhie_train_res_recon_error = []
        randhie_train_res_perplexity = []

        for i in xrange(NUM_TRAINING_UPDATES):
            data = next(iter(randhie_training_loader))
            data = data.to(device)
            randhie_optimizer.zero_grad()

            vq_loss, data_recon, perplexity = randhie_model.forward(data)
            recon_error = F.mse_loss(data_recon, data) / randhie_variance
            loss = recon_error + vq_loss
            loss.backward()

            randhie_optimizer.step()
            
            randhie_train_res_recon_error.append(recon_error.item())
            randhie_train_res_perplexity.append(perplexity.item())

            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(randhie_train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(randhie_train_res_perplexity[-100:]))
                print()
                
        randhie_train_res_recon_error_smooth = savgol_filter(randhie_train_res_recon_error, 201, 7)
        randhie_train_res_perplexity_smooth = savgol_filter(randhie_train_res_perplexity, 201, 7)
        
        # Plot loss
        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(randhie_train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')

        ax = f.add_subplot(1,2,2)
        ax.plot(randhie_train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')
        
        randhie_model.eval()

        # Get a batch of validation data
        randhie_valid_originals = next(iter(randhie_validation_loader))
        randhie_valid_originals = randhie_valid_originals.to(device)

        # Run the batch through the model to get the reconstructions
        randhie_vq_output_eval = randhie_model._pre_vq_conv(randhie_model._encoder(randhie_valid_originals))
        _, randhie_valid_quantize, _, _ = randhie_model._vq_vae(randhie_vq_output_eval)
        randhie_valid_reconstructions = randhie_model._decoder(randhie_valid_quantize)

        # Convert tensors to dataframes
        randhie_valid_originals_df = tensor_to_df(randhie_valid_originals, randhie_columns)
        randhie_valid_reconstructions_df = tensor_to_df(randhie_valid_reconstructions, randhie_columns)

        # Visualization: Compare original and reconstructed values for the first n rows
        n = 5  # Number of rows you want to visualize
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 2*n))

        for i in range(n):
            # Original data visualization
            axes[i, 0].set_title(f'Original Row {i}')
            axes[i, 0].bar(randhie_valid_originals_df.columns, randhie_valid_originals_df.iloc[i], color='blue')
            
            # Reconstructed data visualization
            axes[i, 1].set_title(f'Reconstructed Row {i}')
            axes[i, 1].bar(randhie_valid_reconstructions_df.columns, randhie_valid_reconstructions_df.iloc[i], color='orange')

        plt.tight_layout()
        plt.show()
        
        ### Train VQ VAE for heart
        heart_model = Model(54, NUM_HIDDENS,
                NUM_EMBEDDINGS, EMBEDDING_DIM, 
                COMMITMENT_COST).to(device)
        # Optimizer
        heart_optimizer = optim.Adam(randhie_model.parameters(), lr=LEARNING_RATE, amsgrad=False)
        
        heart_model.train()
        heart_train_res_recon_error = []
        heart_train_res_perplexity = []

        for i in xrange(NUM_TRAINING_UPDATES):
            data = next(iter(heart_training_loader))
            data = data.to(device)
            heart_optimizer.zero_grad()

            vq_loss, data_recon, perplexity = heart_model.forward(data)
            recon_error = F.mse_loss(data_recon, data) / heart_variance
            loss = recon_error + vq_loss
            loss.backward()

            heart_optimizer.step()
            
            heart_train_res_recon_error.append(recon_error.item())
            heart_train_res_perplexity.append(perplexity.item())

            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(heart_train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(heart_train_res_perplexity[-100:]))
                print()
                
        heart_train_res_recon_error_smooth = savgol_filter(heart_train_res_recon_error, 201, 7)
        heart_train_res_perplexity_smooth = savgol_filter(heart_train_res_perplexity, 201, 7)
        
        # Plot loss
        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(heart_train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')

        ax = f.add_subplot(1,2,2)
        ax.plot(heart_train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')
        
        heart_model.eval()
        
        # Get a batch of validation data
        heart_valid_originals = next(iter(heart_validation_loader))
        heart_valid_originals = heart_valid_originals.to(device)

        # Run the batch through the model to get the reconstructions
        heart_vq_output_eval = heart_model._pre_vq_conv(heart_model._encoder(heart_valid_originals))
        _, heart_valid_quantize, _, _ = heart_model._vq_vae(heart_vq_output_eval)
        heart_valid_reconstructions = heart_model._decoder(heart_valid_quantize)

        # Convert tensors to dataframes
        heart_valid_originals_df = tensor_to_df(heart_valid_originals, heart_columns)
        heart_valid_reconstructions_df = tensor_to_df(heart_valid_reconstructions, heart_columns)

        # Visualization: Compare original and reconstructed values for the first n rows
        n = 5  # Number of rows you want to visualize
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 2*n))

        for i in range(n):
            # Original data visualization
            axes[i, 0].set_title(f'Original Row {i}')
            axes[i, 0].bar(heart_valid_originals_df.columns, heart_valid_originals_df.iloc[i], color='blue')
            
            # Reconstructed data visualization
            axes[i, 1].set_title(f'Reconstructed Row {i}')
            axes[i, 1].bar(heart_valid_reconstructions_df.columns, heart_valid_originals_df.iloc[i], color='orange')

        plt.tight_layout()
        plt.show()
        
        return device, randhie_model, heart_model