import os
import sys
import pandas as pd

script_directory = os.path.dirname(os.path.abspath(__file__))
projects_directory = os.path.dirname(script_directory)
sys.path.append(projects_directory)

from Preprocess import raw_dataframe_preprocessor, column_optimizer

def main():
    # randhie dataset path
    randhie_path = os.getcwd()+"/Heart_Attack_Predictor/Datasets/randhie.csv"
    
    # Initialize RANDHIE class instance
    randhie = raw_dataframe_preprocessor.RANDHIE()
    
    # Pre-processed randhie dataset
    randhie_preprocessed = randhie.improved_preprocess(randhie_path)
    
    # heart dataset path
    heart_path = os.getcwd()+"/Heart_Attack_Predictor/Datasets/heart_attack_prediction_dataset.csv"
    
    # Initialize HEART class instance
    heart = raw_dataframe_preprocessor.HEART()
    
    heart_preprocessed = heart.preprocess(heart_path)
    
    column_rearranger = column_optimizer.ColumnRearranger()
    
    # Rearrange columns of the right table such that it aligns most closely with the columns of the left table
    heart_preprocessed_rearranged = column_rearranger.return_optimal_rearrangement(randhie_preprocessed, heart_preprocessed)
    # Test if rearrangement was done correctly
    raw_dataframe_preprocessor.save_dataframe(heart_preprocessed_rearranged, os.getcwd()+"/Heart_Attack_Predictor/Datasets", "heart_preprocessed_rearranged.csv")

if __name__ == "__main__":
    main()