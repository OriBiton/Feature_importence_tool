import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import re


def detect_problem_type(target_column):
    unique_values = target_column.unique()
    if set(unique_values) == {0, 1}:
        return "classification"
    if np.issubdtype(target_column.dtype, np.number):
        return "regression"
    else:
        return "classification"
    


def encoding_and_normalizing(df,target_column):
    df=df.copy()
    target=df[target_column]
    df=df.drop(target_column,axis=1)
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols]=df[categorical_cols].astype(str)
    
    #normalizing
    if len(numeric_cols)>0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    #encoding
    if len(categorical_cols):
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True).astype(int)

    df[target_column]=target
    return df,df.columns


def build_and_train_model(df, problem_type,target_column):
    y=df[target_column]
    X=df.drop(target_column,axis=1)
    X.columns = X.columns.str.replace(" ", "_")
    

    X.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in X.columns]  # החלפת כל תו מיוחד בקו תחתון
    



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trained_features = X_train.columns.tolist()
    best_model = None
    best_score = None
    best_model_name = ""
    results=[]

    if problem_type == "regression":
        models = {
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "LGBMRegressor": LGBMRegressor(),
            "XGBRegressor": XGBRegressor(),
            
        }
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            results.append(f"Model: {model_name}, MSE: {mse:.4f}")
            if best_score is None or mse < best_score:
                best_model, best_score, best_model_name = model, mse, model_name
    else:
        models = {
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "LGBMClassifier": LGBMClassifier()
        }
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results.append(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
            if best_score is None or accuracy > best_score:
                best_model, best_score, best_model_name = model, accuracy, model_name
    results.append(f"Best Model: {best_model_name}")
    if problem_type == "regression":
        results.append(f"Best Model MSE: {best_score:.4f}")
    else:
        results.append(f"Best Model Accuracy: {best_score:.4f}")

    return best_model, trained_features, results,X_train