from flask import jsonify, Flask, request
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

try:
    model = load("poly_reg_model.joblib")
    poly_transformer = load("poly_transformer.joblib")
    print("Model and transformer loaded successfully")
except FileNotFoundError:
    print("File Missing")
    model, poly_transformer = None, None
@app.route("/predict", methods = ["POST"])
def predict():
    if model is None or poly_transformer is None:
        return jsonify({"error": "Model Service Unavailable"})
    try:
        data = request.get_json(force = True)

        feature_value = data['x']
        input_data = np.array([[feature_value]])

        input_poly = poly_transformer.transform(input_data)
        prediction = model.predict(input_poly)[0]

        return jsonify({
            "status": "success",
            "input": feature_value,
            "predicted output": round(prediction, 4)
        })
    except KeyError:
        return jsonify({"error": "Missing required input"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction Failed: {e}"}), 500

if __name__ == "__main__":
    print("Starting Flask API server ...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host = '0.0.0.0', port = port)
