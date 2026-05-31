import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train Support Vector Mechanism
svm = SVC(probability=True)

svm.fit(X_train, y_train)
print("Training completed successfully")

y_pred = svm.predict(X_test)

# Accuracy

print("Accuracy in prediction")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(svm, open("models/svm_model.pkl", "wb"))

print("Model for support vector mechanism is saved")

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))