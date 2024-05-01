from multiprocessing import freeze_support
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

script_directory = os.path.dirname(os.path.abspath(__file__))
projects_directory = os.path.dirname(script_directory)
sys.path.append(projects_directory)

from Vectorize import VAE
from Preprocess import raw_dataframe_preprocessor, column_optimizer
from Predict import predictors as oracle
from Vectorize import Encoder
from HNSW import Row_Matcher

def main():
    # randhie dataset path
    randhie_path = os.getcwd()+"/PVM/Datasets/randhie.csv"
    
    # Initialize RANDHIE class instance
    randhie = raw_dataframe_preprocessor.RANDHIE()
    
    # Pre-processed randhie dataset
    randhie_preprocessed, randhie_X = randhie.improved_preprocess(randhie_path)
    
    """
        The purpose of this research is NOT improving the prediction accuracy of the RANDHIE experiment. It is to extract additional statistically significant
        predictors of quantity demanded for medical care when running a linear regression. This means as long as linearity between a regressor and the target is proven, we do not need to do an out of sample testing for the quantity demanded for medical care itself.
    """
    
    # heart dataset path
    heart_path = os.getcwd()+"/PVM/Datasets/heart_attack_prediction_dataset.csv"
    
    # Initialize HEART class instance
    heart_processor = raw_dataframe_preprocessor.HEART()
    heart_preprocessed_whole, heart_X_whole, heart_y_whole, heart_X_whole_tensor, heart_y_whole_tensor = heart_processor.preprocess(heart_path)
    
    heart_X_whole_with_heartrisk = oracle.run_model_pipeline_and_return_final_heart_predictors(heart_processor, heart_path, heart_X_whole)
    
    ### Column rearranging 
    column_rearranger = column_optimizer.ColumnRearranger()
    
    # Reduce row number of heart table to match that of randhie via bootstrapping
    heart_X = column_rearranger.bootstrap_to_match(randhie_X, heart_X_whole_with_heartrisk)
    
    raw_dataframe_preprocessor.save_dataframe(heart_X, os.getcwd()+"/PVM/Datasets", "heart_preprocessed_X.csv")
    
    # pre-rearrangement
    average_correlation_pre = column_rearranger.compute_average_correlation(randhie_X, heart_X)
    print(f"pre operation average correlation: {average_correlation_pre}")
    
    # Rearrange columns of the right table such that the average correlation between every column i from the left table and every column j from the right table where i=j is maximized
    heart_X_rearranged = column_rearranger.return_optimal_rearrangement(randhie_X, heart_X)
    
    # post-rearrangement
    average_correlation_post = column_rearranger.compute_average_correlation(randhie_X, heart_X_rearranged)
    print(f"post operation average correlation: {average_correlation_post}")
    
    column_rearranger.visualize_comparison(average_correlation_pre, average_correlation_post)
    
    # Update global data
    raw_dataframe_preprocessor.update_heart_final_predictors(heart_X_rearranged, list(heart_X_rearranged.columns))
    
    # Visualize if rearrangement was done correctly
    raw_dataframe_preprocessor.save_dataframe(heart_X_rearranged, os.getcwd()+"/PVM/Datasets", "heart_preprocessed_X_rearranged.csv")
    
    print(f"final heart X rearranged columns: {list(heart_X_rearranged.columns)}")
    
    ### Adding the Vector Encoded Column that summarizes each row's data using VAE
    encoder = Encoder.DataFrameEncoder()
    # Train the models
    # encoder.train_and_assign_models(randhie_preprocessed, heart_X_rearranged, heart_preprocessed_whole)
    # Save the models
    # encoder.save_model(encoder.randhie_model, 'randhie_model.pth')
    # encoder.save_model(encoder.heart_model, 'heart_model.pth')
    
    ## Add a vector encoding column to RANDHIE and HEART dataframes
    
    # HYPERPARAMETERS
    batch_size, num_training_updates, num_hiddens, embedding_dim, learning_rate = VAE.return_hyperparameters()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoded_randhie_df, encoded_heart_df = encoder.load_and_encode_dataframes(randhie_X, heart_X_rearranged, 44, 43, num_hiddens, embedding_dim, device, learning_rate)
    
    raw_dataframe_preprocessor.save_dataframe(encoded_randhie_df, os.getcwd()+"/PVM/Datasets", "randhie_predictors.csv")
    print(f"{encoded_randhie_df.head()}")
    
    raw_dataframe_preprocessor.save_dataframe(encoded_heart_df, os.getcwd()+"/PVM/Datasets", "heart_predictors.csv")
    print(f"{encoded_heart_df.head()}")
    
    ### Probabilistic Vectorized Matching
    
    # randhie_predictors_path = os.getcwd()+"/PVM/Datasets/randhie_predictors.csv"
    # randhie_predictors = encoded_randhie_df
    # pd.read_csv(randhie_predictors_path)
    
    # heart_predictors_path = os.getcwd()+"/PVM/Datasets/heart_predictors.csv"
    # heart_predictors = encoded_heart_df
    # pd.read_csv(heart_predictors_path)

    def match_rows(randhie_df, heart_df):
        row_matcher = Row_Matcher.RowMatcher()
        return row_matcher.retrieve_similar(randhie_df, heart_df)
    
    # Perform row matching and store results
    combined_data = match_rows(encoded_randhie_df, encoded_heart_df)
    
    ### Prepare Data for Final OLS Regression
    final_randhie_regressors, final_heart_regressors, final_randhie_y, final_heart_y = raw_dataframe_preprocessor.return_final_variables()
    combined_data[final_randhie_y] = randhie_preprocessed[final_randhie_y]
    combined_data[final_heart_y] = heart_preprocessed_whole[final_heart_y]

    # Save and display results
    combined_data.to_csv(os.getcwd() + "/PVM/Datasets/merged_predictors.csv")
    print(combined_data.head())
    
    """
        FINAL RESULTS
    """
    oracle.test_research_null(combined_data, combined_data[final_randhie_regressors], combined_data[final_heart_regressors], combined_data[final_randhie_y], combined_data[final_heart_y])
    

if __name__ == "__main__":
    ### Multiprocessing for deep learning
    freeze_support()
    main()