from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def train_local_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = svm.SVC(kernel='linear', probability=True)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model.fit(X_train, y_train)
    return model, scaler

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }

def FedAvg(global_weights, new_weights):
    return (global_weights + new_weights) / 2
