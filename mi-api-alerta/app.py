from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
modelo = joblib.load("modelo_alerta.pkl")

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        pred = modelo.predict(features)[0]
        return jsonify({"prediccion": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()
