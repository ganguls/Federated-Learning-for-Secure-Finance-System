import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
import joblib
import os

def main():
    print("Starting data preprocessing for Federated Learning...")
    
    # Load Lending Club CSV
    print("Loading dataset...")
    df = pd.read_csv("../original_dataset/loan.csv", low_memory=False)
    print(f"Original dataset shape: {df.shape}")
    
    # Drop irrelevant columns
    print("Dropping irrelevant columns...")
    irrelevant_cols = ['id', 'member_id', 'url', 'desc', 'title', 'emp_title', 'zip_code']
    df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns])
    
    # Handle missing values
    print("Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    
    # Create target variable
    print("Creating target variable...")
    df['loan_status_binary'] = (df['loan_status'] == 'Fully Paid').astype(int)
    df = df.drop(columns=['loan_status'])
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    ordinal_cols = ['emp_length', 'sub_grade']
    for col in ordinal_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, f"{col}_encoder.pkl")
    
    # One-hot encoding for nominal columns
    nominal_cols = ['grade', 'home_ownership', 'purpose', 'addr_state']
    nominal_cols = [col for col in nominal_cols if col in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Remove columns with too many unique values or low variance
    print("Feature selection...")
    for col in df.columns:
        if col != 'loan_status_binary':
            if df[col].nunique() > 1000:  # Too many unique values
                df = df.drop(columns=[col])
            elif df[col].dtype in ['object']:
                df = df.drop(columns=[col])
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Target variable distribution:")
    print(df['loan_status_binary'].value_counts(normalize=True))
    
    # Split data into clients for federated learning
    print("Splitting data into clients...")
    n_clients = 10
    client_data = np.array_split(df, n_clients)
    
    # Ensure FL_clients directory exists
    os.makedirs("FL_clients", exist_ok=True)
    
    # Save client data
    for i, client_df in enumerate(client_data, 1):
        client_df.to_csv(f"FL_clients/client_{i}.csv", index=False)
        print(f"Client {i}: {client_df.shape[0]} samples, {client_df.shape[1]} features")
    
    # Save the full processed dataset
    df.to_csv("processed_full_dataset.csv", index=False)
    
    # Save feature names for later use
    feature_names = [col for col in df.columns if col != 'loan_status_binary']
    joblib.dump(feature_names, "feature_names.pkl")
    
    print("\nData preprocessing completed!")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(feature_names)}")
    print(f"Positive class ratio: {df['loan_status_binary'].mean():.3f}")

if __name__ == "__main__":
    main()
