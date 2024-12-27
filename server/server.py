from flask import Flask, request, jsonify
import numpy as np
from utils.utils import FedAvg

app = Flask(__name__)
global_model_weights = None

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    global global_model_weights
    try:
        # Validate input data
        if 'weights' not in request.json:
            return jsonify({"status": "error", "message": "No weights provided."}), 400
        
        weights = request.json['weights']

        # Convert weights to a numpy array for consistency
        weights_array = np.array(weights)

        # Check if weights_array is valid
        if weights_array.ndim == 1:
            weights_array = weights_array.reshape(1, -1)  # Reshape if it's a 1D array

        # Initialize or average weights
        if global_model_weights is None:
            global_model_weights = weights_array
        else:
            global_model_weights = FedAvg(global_model_weights, weights_array)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/download_weights', methods=['GET'])
def download_weights():
    if global_model_weights is None:
        return jsonify({"weights": []}), 200  # Return an empty list if no weights are available
    return jsonify({"weights": global_model_weights.flatten().tolist()}), 200  # Flatten for consistent output

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
