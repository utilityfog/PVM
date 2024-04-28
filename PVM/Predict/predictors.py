from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import statsmodels.api as sm
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision

from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms, utils

from xgboost import XGBClassifier

from Preprocess import raw_dataframe_preprocessor
from Vectorize import VAE

def test_research_null(combined_data: pd.DataFrame, final_randhie_regressors, final_heart_regressors, final_randhie_y, final_heart_y):
    """
        Summary: The results show that 
    """
    
    predictors = final_randhie_regressors + ['Heart_Attack_Risk_Predicted', 'Stress Level', 'Sedentary Hours Per Day', 'Obesity_1', 'Cholesterol']
    print(f"Using predictors: {predictors}")

    regression_results = {}

    for target in final_randhie_y.columns:
        print(f"Running regression for target: {target}")

        # IN Sample OLS regression to determine whether our new predictors have a linear correlation with quantity demanded for medical care, ceteribus paribus including effective price
        X = combined_data[predictors]
        y = final_randhie_y[target]
        X_with_const = sm.add_constant(X)  # Add constant for the intercept term

        model = sm.OLS(y, X_with_const)
        results = model.fit()

        y_pred = results.predict(X_with_const)
        mse = mean_squared_error(y, y_pred)

        regression_results[target] = {
            'Summary': results.summary(),
            'MSE': mse
        }

    for target, data in regression_results.items():
        print(f"Results for {target}:")
        print(data['Summary'])
        print(f"Mean Squared Error: {data['MSE']}")
    
def run_model_pipeline_and_return_final_heart_predictors(heart_processor: raw_dataframe_preprocessor.HEART, heart_path: str, heart_X: pd.DataFrame, heart_y: pd.DataFrame=None) -> pd.DataFrame:
    # Define a dictionary to hold all the model prediction functions
    models = {
        'Lasso Logistic Regression': lasso_logistic_predict,
        'Elastic Net Logistic Regression': elastic_net_logistic_predict,
        'Support Vector Machine (RBF)': svm_rbf_predict,
        'XGBoost': xgboost_predict,
        'Simple Neural Network': simple_NN_predict,
        'Transformer': transformer_predict
    }

    # Results dictionary to track model performance
    results = {}
    model_objects = {}

    # Run each model and collect results
    for name, predict_func in models.items():
        print(f"Running {name}...")
        ensemble_models, metric = cross_validate_and_ensemble(predict_func, heart_path, heart_processor)
        if isinstance(metric, dict):  # For models that return metrics as dict
            results[name] = metric
        else:
            results[name] = {'AUC': metric}
        print(f"{name} completed. AUC: {metric}")
        model_objects[name] = ensemble_models

    # Find the model with the best performance based on a chosen metric
    # Assuming all models return 'AUC' as their metric for simplicity
    best_models_id = max(results, key=lambda x: results[x].get('auc', 0))
    # best_model_name = max(results, key=results.get)
    best_model_metrics = results[best_models_id]
    print(f"Best models are {best_models_id} with AUC: {best_model_metrics}")
    # Find the model with the best performance based on AUC
    best_models = model_objects[best_models_id] # best_models: list(model)
    print(f"Best models are {best_models_id}: {results[best_models_id]}")
    
    # Ensemble the predictions from the best model list
    predictions = [model.predict_proba(heart_X)[:, 1] for model in best_models]
    mean_predictions = np.mean(predictions, axis=0)

    # Add predictions as a new column to heart_X
    heart_X['Heart_Attack_Risk_Predicted'] = mean_predictions
    
    raw_dataframe_preprocessor.update_heart_final_predictors(heart_X, list(heart_X.columns))

    return heart_X

def calculate_rmse(model, data_loader):
    model.eval()  # Set model to evaluation mode
    predictions, actuals = [], []
    with torch.no_grad():
        for data, targets in data_loader:
            outputs = model(data)
            predicted_classes = torch.softmax(outputs, dim=1).max(dim=1)[1]
            predictions.extend(predicted_classes.numpy())
            actuals.extend(targets.numpy())
    return np.sqrt(mean_squared_error(actuals, predictions))

def cross_validate_and_ensemble(predict_func, heart_path: str, processor: raw_dataframe_preprocessor.HEART, n_splits=5):
    """
    Perform K-fold cross-validation using a custom prediction function that returns a model and its AUC score for each fold.
    
    Args:
    - predict_func (callable): A function that accepts X_train, y_train, X_test, y_test and returns a model and AUC.
    - X (pd.DataFrame): Feature data.
    - y (pd.Series): Target data.
    - n_splits (int): Number of folds for K-Fold cross-validation.
    
    Returns:
    - ensemble_models: A list containing trained models on each fold.
    - mean_auc: Mean AUC score across all folds to evaluate performance.
    """
    kf = raw_dataframe_preprocessor.OOS(n_splits=n_splits, shuffle=True, random_state=42)
    ensemble_models = []
    auc_scores = []
    
    # k=5 Cross Validation
    heart_df = pd.read_csv(heart_path)
    i = 0
    for train_fold_index, test_fold_index in kf.split(heart_df):
        heart_df_train_fold, heart_df_test_fold = heart_df.iloc[train_fold_index], heart_df.iloc[test_fold_index]
        
        if predict_func.__name__ in ['simple_NN_predict', 'transformer_predict']:
            # These functions require tensor data
            _, heart_X_train_fold, heart_y_train_fold, heart_X_train_fold_tensor, heart_y_train_fold_tensor = processor.preprocess(heart_df_train_fold)
            
            # Test here if heart_X_train_fold has variables that may cause perfect multicollinearity
        
            _, heart_X_test_fold, heart_y_test_fold, heart_X_test_fold_tensor, heart_y_test_fold_tensor = processor.preprocess(heart_df_test_fold)

            model, auc = predict_func(heart_X_train_fold, heart_y_train_fold, heart_X_test_fold, heart_y_test_fold, heart_X_train_fold_tensor, heart_y_train_fold_tensor, heart_X_test_fold_tensor, heart_y_test_fold_tensor)
        else:
            # Other functions use just DataFrame input
            _, heart_X_train_fold, heart_y_train_fold, _, _ = processor.preprocess(heart_df_train_fold)
        
            _, heart_X_test_fold, heart_y_test_fold, _, _ = processor.preprocess(heart_df_test_fold)

            model, auc = predict_func(heart_X_train_fold, heart_y_train_fold, heart_X_test_fold, heart_y_test_fold, i)
        i += 1

        # Store the model from the current fold
        ensemble_models.append(model)

        # Store the AUC score
        auc_scores.append(auc.get('auc'))

    # Calculate mean AUC across all folds
    mean_auc = np.mean(auc_scores)

    return ensemble_models, mean_auc

def lasso_logistic_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, fold_index):
    print("LASSO Logistic Prediction!")
    # Initialize Logistic Regression with L1 penalty (Lasso)
    model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)
    model.fit(X_train, np.ravel(y_train,order='C'))
    
    # Predict probabilities on the test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)

    results = {
        'auc': auc
    }

    # Plot coefficients of the model
    plt.figure(figsize=(10, 5))
    plt.plot(model.coef_.flatten(), marker='o', linestyle='none')
    plt.title('Coefficients of the Lasso Model')
    plt.xlabel('Features Index')
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    #plt.show()
    
    plt.savefig(f'./PVM/Plots/lasso_coef{fold_index}.png')
    plt.close()
    
    return model, results

def elastic_net_logistic_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, fold_index):
    print("Elastic Net Logistic Prediction!")
    # Setup ElasticNetCV with more iterations
    model = ElasticNetCV(cv=5, l1_ratio=np.linspace(0.01, 1, 100), alphas=np.logspace(-6, 2, 100), 
                         max_iter=500, tol=0.0001, random_state=42)
    
    model.fit(X_train, np.ravel(y_train,order='C'))
    
    # Predicting on test set
    y_pred = model.predict(X_test)
    
    # Compute AUC
    auc = roc_auc_score(y_test, y_pred)
    
    results = {
        'auc': auc,
        'best_lambda': model.alpha_,
        'best_l1_ratio': model.l1_ratio_
    }

    # Plot coefficients of the best model
    plt.figure(figsize=(10, 5))
    plt.plot(model.coef_, marker='o', linestyle='none')
    plt.title('Coefficients of the Best Model')
    plt.xlabel('Features Index')
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./PVM/Plots/elastic_coef{fold_index}.png') 
    plt.close()
    
    return model, results

def svm_rbf_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, fold_index):
    print("SVM with RBF Kernel Prediction!")
    # SVM with RBF kernel
    svm_model = SVC(kernel='rbf', gamma='scale', probability=True)
    svm_model.fit(X_train, np.ravel(y_train,order='C'))

    # Predict probabilities to compute AUC
    y_pred_probs = svm_model.predict_proba(X_test)[:, 1]

    # Compute AUC
    auc = roc_auc_score(y_test, y_pred_probs)

    # Results dictionary
    results = {
        'auc': auc
    }

    return svm_model, results
    
def xgboost_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, fold_index):
    print("XGBoost Prediction!")
    # Setup XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, np.ravel(y_train,order='C'))
    
    # Predict probabilities for AUC calculation
    y_pred_probs = model.predict_proba(X_test)[:, 1]

    # Compute AUC
    auc = roc_auc_score(y_test, y_pred_probs)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(ticks=range(len(X_train.columns)), labels=X_train.columns, rotation=90)
    plt.tight_layout()
    # plt.show()
    
    plt.savefig(f'./PVM/Plots/XGBoost_Prediction{fold_index}.png') 
    plt.close()
    
    # Results dictionary
    results = {
        'auc': auc
    }

    return model, results
    
def simple_NN_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor):
    print("Simple Neural Network Prediction!")
    # Retrieve hyperparameters
    batch_size, num_training_updates, num_hiddens, embedding_dim, learning_rate = VAE.return_hyperparameters()

    # Data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.float().view(-1, 1))  # Ensure target is [batch_size, 1]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the SimpleNN model
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, num_hiddens)
            self.fc2 = nn.Linear(num_hiddens, num_hiddens // 2)
            self.fc3 = nn.Linear(num_hiddens // 2, 1)  # Output one value
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            return torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification

    model = SimpleNN(X_train_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    model.train()
    for epoch in range(num_training_updates):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        auc_score = roc_auc_score(y_test_tensor.view(-1, 1), predictions)  # Matching target shape [batch_size, 1]

    return model, {'auc': auc_score}
    
def transformer_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor):
    print("Transformer Prediction!")
    # Retrieve hyperparameters
    batch_size, num_training_updates, num_hiddens, embedding_dim, learning_rate = VAE.return_hyperparameters()
    # print(f"X_train columns: {X_train.columns}")
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Obsolete
    # X_train_tensor = X_train_tensor.to(device)
    # y_train_tensor = y_train_tensor.to(device).float().view(-1, 1)
    # X_test_tensor = X_test_tensor.to(device)
    # y_test_tensor = y_test_tensor.to(device).float().view(-1, 1)
    
    # Data tensors should be sent to the appropriate device and reshaped if necessary
    X_train_tensor = X_train_tensor.cpu()
    y_train_tensor = y_train_tensor.cpu().float().view(-1, 1)
    X_test_tensor = X_test_tensor.cpu()
    y_test_tensor = y_test_tensor.cpu().float().view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if __name__ == '__main__':
        freeze_support()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )
    
    # print("Fully read DataLoader!")

    # Transformer Model
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(TransformerModel, self).__init__()
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=14, dropout=0.1, batch_first=True, device=device
            )
            self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
            self.classifier = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            x = x.unsqueeze(0)  # Add batch dimension
            x = self.transformer_encoder(x)
            x = torch.mean(x, dim=0)  # Average pooling along the sequence dimension
            return torch.sigmoid(self.classifier(x))  # Use sigmoid for binary classification

    model = TransformerModel(X_train_tensor.shape[1], 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    model.train()
    for epoch in range(num_training_updates):
        for data, target in train_loader:
            # Move data to the device each batch
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, target)  # Ensure output and target shapes are aligned
            print(f"transformer training loss: {loss}")
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        predictions = model.forward(X_test_tensor)
        predictions_unloaded = predictions.cpu()
        y_test_unloaded = y_test_tensor.cpu()
        auc_score = roc_auc_score(y_test_unloaded, predictions_unloaded)

    return model, {'auc': auc_score}