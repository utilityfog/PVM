class heart_attack_prediction_dataset:
    '''
    This class is made to work with the heart_attack_predictio_dataset.csv dataset.
    '''
    def load_clean(self, df_path):
        '''
        Load the dataset and clean it up
        
        CleaningDecisions:
        1. Seperate Systolic and Diastolic blood pressure into seperate columns 
        2. Create an interaction term between systolic and diastolic.
        3. Drop 'Sex_Male', 'Diet_Average', 'Country_Argentina' to prevent multicolinearity.
        4. Set categorical variables to category type.
        '''
        
        import pandas as pd
        import numpy as np

        df = pd.read_csv('heart_attack_prediction_dataset.csv')
        
        # Switch income to log income.
        Income = df['Income']
        df['Log Income'] = np.log(Income)

        # Set Categorical Variables
        categorical_variables = ['Sex', 'Diet', 'Country']

        for col in categorical_variables:
                df[col] = df[col].astype('category')

        ### One hot encode the categorical variables
        df = pd.get_dummies(df, columns=['Sex', 'Diet','Country'])
        # Drop to prevent multicolinearity with binary variables
        df = df.drop(['Sex_Male', 'Diet_Average', 'Country_Argentina'], axis=1)

        ### Seperate systolic and diastolic blood pressure into their own variables and create an interaction term.
        # Split the 'Blood Pressure' column into 'Systolic' and 'Diastolic'
        df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

        # Drop for no need
        df = df.drop(['Patient ID', 'Blood Pressure', 'Income', 'Continent', 'Hemisphere'], axis=1)

        # Create interaction term by multiplying Systolic and Diastolic pressures
        df['BP_Interaction'] = df['Systolic'] * df['Diastolic']
        
        return df
    
    def normalize(self, df):
        '''
        Normalize the features in the dataframe.
        '''
        df - self.scaler.fit_transform(df)

        return df
        
    def split(self, X, y):
        '''
        Split the data into training and testing sets.

        Returns x_train, y_train, x_test, y_test, train_loader, val_loader
        '''
        #print(list(df.columns))
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import torch
        import torch.nn as nn
        import numpy as np
        from sklearn.metrics import accuracy_score
        from torch.utils.data import DataLoader, TensorDataset
        import pandas as pd  

        # Separate the target variable (y) and the features (X)
        X = df.drop('Heart Attack Risk', axis=1)
        y = df['Heart Attack Risk']

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        # Convert from numpy arrays or pandas DataFrames to PyTorch Tensors
        X_train = torch.tensor(X_train.values.astype(np.float32)) if isinstance(X_train, pd.DataFrame) else torch.tensor(X_train.astype(np.float32))
        y_train = torch.tensor(y_train.values.astype(np.int64)) if isinstance(y_train, pd.Series) else torch.tensor(y_train.astype(np.int64))
        X_test = torch.tensor(X_test.values.astype(np.float32)) if isinstance(X_test, pd.DataFrame) else torch.tensor(X_test.astype(np.float32))
        y_test = torch.tensor(y_test.values.astype(np.int64)) if isinstance(y_test, pd.Series) else torch.tensor(y_test.astype(np.int64))

        # Define datasets and DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        print("Training dataset shape:", X_train.shape, y_train.shape)
        print("Validation dataset shape:", X_test.shape, y_test.shape)
        print("Training dataset type:", type(X_train), type(y_train))
        print("Validation dataset type:", type(X_test), type(y_test))
        print('')
        print("Object type of current x and y variables:", type(X_train))
        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)

        return X_train, X_test, y_train, y_test, train_loader, val_loader