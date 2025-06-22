import joblib
import pandas as pd

def predict(input_data, model_path="model.pkl"):
    print(f"Making prediction for input: {input_data}")
    try:
        model = joblib.load(model_path)
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        print(f"Prediction made: {prediction}")
        return prediction
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Cannot make prediction. Returning default.")
        return "No model found"
    except Exception as e:
        print(f"Error during prediction: {e}. Returning default.")
        return "Prediction error"
