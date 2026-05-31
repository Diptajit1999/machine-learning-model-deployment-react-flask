import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset

data = pd.read_csv("dataset/Social_Network_Ads.csv")

print("dataset Loaded")


# Features

X = data[['Age', 'EstimatedSalary']]

# Target

y = data['Purchased']


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Training with Decision Tree Algorithm
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

# Accuracy
print("Accuracy in Prediction")
print("Accuracy:", accuracy_score(y_test, y_pred))


# Save Model
pickle.dump(dt, open("models/decision_tree_model.pkl", "wb"))

print("Model Saved")