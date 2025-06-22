import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    print(f"Loading and preprocessing data from {file_path}...")
    try:
        df = pd.read_csv(file_path, sep=';')

        # --- Ensure O2 is numeric for target creation ---
        df['O2'] = pd.to_numeric(df['O2'], errors='coerce')
        # Fill NaN O2 values, e.g., with mean, before creating target. Use 0 if O2 is completely empty.
        df['O2'] = df['O2'].fillna(df['O2'].mean() if not df['O2'].empty else 0)

        # --- Always create 'is_good_quality' as the target based on O2 threshold ---
        # Define the target variable: 1 if O2 >= 8.0, else 0
        df['is_good_quality'] = (df['O2'] >= 8.0).astype(int)
        print("Created 'is_good_quality' target based on O2 threshold (O2 >= 8.0).")

        # Define features (X) and target (y)
        y = df['is_good_quality']
        # Drop 'id', 'date', and the newly created target column from features
        # 'errors='ignore'' prevents an error if a column is already dropped or not found
        X = df.drop(columns=['id', 'date', 'is_good_quality'], errors='ignore')

        # Convert all remaining feature columns in X to numeric, coercing errors to NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Handle missing values in numerical features (X) by filling with the mean of each column
        # Ensure only numeric columns are considered for mean calculation
        numeric_cols_X = X.select_dtypes(include=np.number).columns
        X[numeric_cols_X] = X[numeric_cols_X].fillna(X[numeric_cols_X].mean())

        print("Data loaded and preprocessed.")
        return X, y
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Returning dummy data for execution.")
        # Fallback dummy data, ensuring it's numeric and has a consistent number of features (9 for this dataset)
        X_dummy = pd.DataFrame(np.random.rand(10, 9), columns=[f'feature_{i}' for i in range(9)])
        y_dummy = pd.Series(np.random.randint(0, 2, 10))
        return X_dummy, y_dummy
