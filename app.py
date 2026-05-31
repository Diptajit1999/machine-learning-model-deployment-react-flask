from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import json
import os

app = Flask(__name__)

CORS(
    app,
    origins=[
        "https://machine-learning-model-deployment-frontend.onrender.com/predict"
    ]
)
# Load models
models = {
    "knn": pickle.load(open("models/knn_model.pkl", "rb")),
    "naive_bayes": pickle.load(open("models/naive_bayes_model.pkl", "rb")),
    "svm": pickle.load(open("models/svm_model.pkl", "rb")),
    "decision_tree": pickle.load(open("models/decision_tree_model.pkl", "rb")),
    "random_forest": pickle.load(open("models/random_forest_model.pkl", "rb"))
}

# Load scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Load accuracies (IMPORTANT)
with open("models/accuracies.json", "r") as f:
    MODEL_ACCURACIES = json.load(f)

@app.route('/')
def home():
    return jsonify({
        "status":"API is running",
        "message": "Social Network Ads ML Prediction API is running"
    })

@app.route('/predict', methods=['POST'])
def predict():

    # Get input from Postman
    data = request.get_json()

    age = int(data['age'])
    salary = int(data['salary'])
    selected_model = data['model']

    
# Validation
    if selected_model not in models:
        return jsonify({
            "error": "Invalid model selected"
            }), 400


    # Create dataframe
    sample = pd.DataFrame(
        [[age, salary]],
        columns=['Age', 'EstimatedSalary']
    )


    # Select model
    model = models[selected_model]

    # Scale if required
    if selected_model in ["knn", "naive_bayes", "svm"]:
        sample = scaler.transform(sample)

    # Prediction
    prediction = model.predict(sample)[0]

    result = "Purchased" if prediction == 1 else "Not Purchased"

    # Get accuracy
    accuracy = MODEL_ACCURACIES[selected_model]

    # Optional probability (if supported)
    confidence = None
    try:
        confidence = round(model.predict_proba(sample)[0][1] * 100, 2)
    except Exception as e:
        print(e)



    

    # print(type(model))
    # print(hasattr(model, "predict_proba"))

    # Return JSON response
    return jsonify({
        "model": selected_model,
        "prediction": result,
        "accuracy": f"{accuracy}%",
        "confidence": f"{confidence}%" if confidence else "Not available"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    