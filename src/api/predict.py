import pickle
from fastapi import FastAPI

from src.helpers.input_processing import preprocess_input

app = FastAPI()

# Load the saved model
with open('../../train.pkl', 'rb') as file:
    model_bundle = pickle.load(file)

# Extract the model and threshold
model = model_bundle['model']
threshold = model_bundle['threshold']

print("Model loaded successfully!")
print("Threshold:", threshold)

@app.get("/")
def home():
    return {"message": "Credit Default Prediction API"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/predict")
def predict(json_data: dict, currency_3_letters: str = 'TWD'):

    df = preprocess_input(json_data, currency_3_letters)
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4)
    }