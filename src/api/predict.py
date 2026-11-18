import pickle
from fastapi import FastAPI

from src.helpers.input_processing import preprocess_input

VERBOSE = True

app = FastAPI()

# Load the saved model

from src.config.constants import PICKLE_FILENAME
from src.config.paths import PROJECT_ROOT_DIR

with open(str(PROJECT_ROOT_DIR / PICKLE_FILENAME), 'rb') as file:
    model_bundle = pickle.load(file)

# Extract the model and threshold
model = model_bundle['model']
threshold = model_bundle['threshold']

if VERBOSE:
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