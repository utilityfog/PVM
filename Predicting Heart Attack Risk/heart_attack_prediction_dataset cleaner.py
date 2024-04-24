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
        '''
        #print(list(df.columns))
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Separate the target variable (y) and the features (X)
        X = df.drop('Heart Attack Risk', axis=1)
        y = df['Heart Attack Risk']

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        print("Object type of current x and y variables:", type(X_train))
        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)
        return X_train, X_test, y_train, y_test