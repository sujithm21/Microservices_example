import requests
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lime import lime_tabular
import shap
from utils.utils import train_local_model, evaluate_model

SERVER_URL = "http://server:5000"

# Load and prepare dataset
data = pd.read_csv('/data/merged_shuffled_NetworkData_20000.csv')

# Check for non-numeric values and convert to numeric
data['is_mal'] = pd.to_numeric(data['is_mal'], errors='coerce')

# Drop rows with NaN values
data.dropna(inplace=True)

# Specify the features to use
features = ['flow_duration', 'Header_Length', 'Source Port', 'Destination Port', 'Protocol Type', 'Duration', 
            'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 
            'ack_flag_number', 'urg_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 
            'fin_count', 'urg_count', 'rst_count', 'max_duration', 'min_duration', 'sum_duration', 'average_duration', 
            'std_duration', 'CoAP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 
            'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 
            'MAC', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight', 'DS status', 'Fragments', 'Sequence number', 
            'Protocol Version', 'flow_idle_time', 'flow_active_time']

# Split features and target variable
X = data[features]  # Use the specified features
y = data['is_mal']

# Train local model
model, scaler = train_local_model(X, y)
local_weights = model.coef_.flatten().tolist()  # Flattening in case of multi-dimensional weights

# Send local weights to server
requests.post(f"{SERVER_URL}/upload_weights", json={"weights": local_weights})

# Download global weights
response = requests.get(f"{SERVER_URL}/download_weights")
global_weights = np.array(response.json()['weights'])

# Create a new model with global weights
new_model = svm.SVC(kernel='linear', probability=True)

# Check if the new model's coef_ has been initialized
try:
    # Check if global weights have the correct shape
    if global_weights.shape[0] == new_model.coef_.shape[1]:
        # Manually set the weights using the global weights
        new_model.coef_ = np.array(global_weights).reshape(new_model.coef_.shape)
    else:
        print("Global weights shape mismatch!")

except AttributeError:
    # Initialize coef_ if it hasn't been created yet
    new_model.fit(X, y)  # Fit the model first to create coef_ if it's not already initialized
    new_model.coef_ = np.array(global_weights).reshape(new_model.coef_.shape)

# If the model is fitted, evaluate it
if hasattr(new_model, 'coef_'):
    # Evaluate the model
    metrics = evaluate_model(new_model, scaler.transform(X), y)
    
    # Print evaluation metrics
    accuracy = accuracy_score(y, new_model.predict(scaler.transform(X)))
    precision = precision_score(y, new_model.predict(scaler.transform(X)), zero_division=0)
    recall = recall_score(y, new_model.predict(scaler.transform(X)), zero_division=0)
    f1 = f1_score(y, new_model.predict(scaler.transform(X)), zero_division=0)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Explain predictions using LIME
    explainer = lime_tabular.LimeTabularExplainer(
        scaler.transform(X),
        feature_names=X.columns.tolist(),
        class_names=['Not Malicious', 'Malicious'],
        mode='classification'
    )

    for i in range(5):  # Display explanations for 5 samples
        exp = explainer.explain_instance(
            scaler.transform(X.iloc[i].values.reshape(1, -1)), 
            new_model.predict_proba
        )
        print(f"LIME Explanation for instance {i}:", exp.as_list())

    # Explain predictions using SHAP
    shap_values = shap.KernelExplainer(new_model.predict_proba, scaler.transform(X))
    shap_values_results = shap_values.shap_values(scaler.transform(X))
    shap.summary_plot(shap_values_results, scaler.transform(X), feature_names=X.columns)

else:
    print("Model has not been fitted properly.")
