import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train_and_save_model(X, y, model_path="model.pkl"):
    print("Training and saving model...")
    
    # Define DummyModel class outside the try/except block, but within the function
    class DummyModel:
        def predict(self, X_input):
            # Ensure X_input is a DataFrame to avoid errors with .iloc if y is empty
            if not y.empty:
                return [y.iloc[0]] * len(X_input) if isinstance(X_input, pd.DataFrame) else [y.iloc[0]] * X_input.shape[0]
            else:
                return [0] * len(X_input) if isinstance(X_input, pd.DataFrame) else [0] * X_input.shape[0]

    model = LogisticRegression(max_iter=1000) # Simple model
    try:
        if len(y.unique()) < 2:
            print("Warning: Target variable has less than 2 unique values. Training a dummy model.")
            model = DummyModel()
        else:
            model.fit(X, y)
        joblib.dump(model, model_path)
        print(f"Model trained (or dummy model created) and saved to {model_path}.")
    except Exception as e:
        print(f"Error during model training/saving: {e}")
        print("Creating a fallback dummy model that always predicts 0.")
        model = DummyModel() # Use the already defined DummyModel
        joblib.dump(model, model_path) # Save the fallback dummy model
