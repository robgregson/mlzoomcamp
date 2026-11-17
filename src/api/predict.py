import pickle
from fastapi import FastAPI

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