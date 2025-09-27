from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved model
with open("HeartDiseaseModel.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        features = [
            float(request.args.get("age")),
            float(request.args.get("sex")),
            float(request.args.get("cp")),
            float(request.args.get("trestbps")),
            float(request.args.get("chol")),
            float(request.args.get("fbs")),
            float(request.args.get("restecg")),
            float(request.args.get("thalach")),
            float(request.args.get("exang")),
            float(request.args.get("oldpeak")),
            float(request.args.get("slope")),
            float(request.args.get("ca")),
            float(request.args.get("thal")),
        ]

        # Convert to numpy array and reshape for prediction
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify({
            "prediction": int(prediction),
            "message": "Heart disease detected" if prediction == 1 else "No heart disease detected"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
