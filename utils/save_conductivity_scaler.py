import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def regenerate_conductivity_scaler():
    train_file = 'data/train_PE_I.csv'
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found.")
        return

    print(f"Loading {train_file} to regenerate scaler...")
    # PE_I files usually don't have headers in this project based on Downstream.py logic
    df = pd.read_csv(train_file, header=None)
    
    # Target is in the second column (index 1)
    target = df.iloc[:, 1].values.reshape(-1, 1)
    
    scaler = StandardScaler()
    scaler.fit(target)
    
    output_path = 'ckpt/scaler_conductivity.joblib'
    joblib.dump(scaler, output_path)
    print(f"Successfully saved scaler to {output_path}")
    print(f"Mean: {scaler.mean_[0]}, Scale: {scaler.scale_[0]}")

if __name__ == "__main__":
    regenerate_conductivity_scaler()
