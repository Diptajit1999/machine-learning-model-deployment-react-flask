import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset/Social_Network_Ads.csv")

print("Dataset Loaded")

# Features
X = data[['Age', 'EstimatedSalary']]

# Target
y = data['Purchased']


# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training Random Forest Algorithm
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Accuracy
print("Accuracy in Prediction")
print("Accuracy:", accuracy_score(y_test, y_pred))


# Save Model
pickle.dump(rf, open("models/random_forest_model.pkl", "wb"))

print("Model of Random Forest Saved")