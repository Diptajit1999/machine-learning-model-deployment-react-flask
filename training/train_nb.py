import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/Social_Network_Ads.csv")

# Features
X = data[['Age', 'EstimatedSalary']]

# Target
y = data['Purchased']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Naive Bayes
nb = GaussianNB()

print("Dataset Loaded")

nb.fit(X_train, y_train)

print("Training completed successfully")

# Accuracy
y_pred = nb.predict(X_test)

print("Accuracy in prediction")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(nb, open("models/naive_bayes_model.pkl", "wb"))

print("Model saved as models/naive_bayes_model.pkl")

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Scaler saved as models/scaler.pkl")