from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

def train_local_model(X, y):
    """
    Train a local SVM model using the provided features and labels.
    
    Parameters:
        X (np.ndarray or pd.DataFrame): Features for training.
        y (np.ndarray or pd.Series): Labels for training.

    Returns:
        model: Trained SVM model.
        scaler: Fitted StandardScaler object.
    """
    # Convert X to DataFrame if it isn't already
    X = pd.DataFrame(X)

    # Drop any non-numeric columns (if necessary)
    X_numeric = X.select_dtypes(include=[np.number])

    # Scale the numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and fit the SVM model
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    return model, scaler

def evaluate_model(model, scaler, X, y):
    """
    Evaluate the trained model using the provided features and labels.
    
    Parameters:
        model: Trained SVM model.
        scaler: StandardScaler object used for scaling.
        X (np.ndarray or pd.DataFrame): Features for evaluation.
        y (np.ndarray or pd.Series): Labels for evaluation.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Convert X to DataFrame if it isn't already
    X = pd.DataFrame(X)

    # Scale the features using the same scaler
    X_scaled = scaler.transform(X.select_dtypes(include=[np.number]))

    # Make predictions
    y_pred = model.predict(X_scaled)

    # Return evaluation metrics
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average='weighted'),
        "recall": recall_score(y, y_pred, average='weighted'),
        "f1_score": f1_score(y, y_pred, average='weighted')
    }

def FedAvg(global_weights, new_weights):
    """
    Federated Averaging function to combine global and new weights.

    Parameters:
        global_weights: Weights from the global model.
        new_weights: Weights from the local model.

    Returns:
        np.ndarray: Averaged weights.
    """
    return (global_weights + new_weights) / 2

# Example usage
if __name__ == "__main__":
    # Generate example data
    X = np.random.rand(100, 5)  # Replace with your actual feature data
    y = np.random.randint(0, 2, size=(100,))  # Replace with your actual labels

    # Train the model
    model, scaler = train_local_model(X, y)

    # Evaluate the model
    metrics = evaluate_model(model, scaler, X, y)
    print("Evaluation Metrics:")
    print(metrics)
