import os
import sys
import pandas as pd

script_directory = os.path.dirname(os.path.abspath(__file__))
projects_directory = os.path.dirname(script_directory)
sys.path.append(projects_directory)

from Preprocess import raw_dataframe_preprocessor

def main():
    # randhie dataset path
    randhie_path = os.getcwd()+"/Heart_Attack_Predictor/Datasets/randhie.csv"
    
    # Initialize RANDHIE class instance
    randhie = raw_dataframe_preprocessor.RANDHIE()
    
    # Pre-processed randhie dataset
    randhie_preprocessed = randhie.improved_preprocess(randhie_path)
    
    print(f"preprocessed rand hie dataset: {randhie_preprocessed.head()}")

if __name__ == "__main__":
    main()