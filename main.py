from src.preprocess import load_and_preprocess_data
from src.train_model import train_and_save_model
from src.predict import predict

def main():
    # Load and preprocess
    X, y = load_and_preprocess_data("data/water_quality.csv") # <--- THIS LINE IS NOW CORRECTED

    # Train model and save
    train_and_save_model(X, y)

    # Predict on a sample input
    sample_input = X.iloc[0].to_dict()
    prediction = predict(sample_input)
    print("Sample input:", sample_input)
    print("Predicted output:", prediction)

if __name__ == "__main__":
    main()
