import pickle
from datetime import datetime
import pandas as pd

# Traduce el código de predicción a tipo de alerta
def interpretar_alerta(codigo):
    return {
        1: "leve",
        2: "grave",
        3: "critica"
    }.get(codigo, "desconocida")

# Predice alerta a partir del JSON de entrada
def predecir_alerta(json_input):
    with open("modelo_completo.lpk", "rb") as f:
        data = pickle.load(f)

    modelo = data["modelo"]
    le = data["label_encoders"]
    columnas = modelo.feature_names_in_.tolist()

    entrada = json_input.copy()

    # Codificación de variables categóricas
    entrada["dia_semana"] = le["dia_semana"].transform([entrada["dia_semana"]])[0]
    entrada["clima"] = le["clima"].transform([entrada["clima"]])[0]
    entrada["ruta"] = le["ruta"].transform([entrada["ruta"]])[0]
    entrada["tramo"] = le["tramo"].transform([entrada["tramo"]])[0]

    X_nuevo = pd.DataFrame([entrada])[columnas]
    pred = modelo.predict(X_nuevo)[0]

    return {
        "n_vehiculo": json_input["n_vehiculo"],
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "estado_velocidad": int(pred),
        "tipo_alerta": interpretar_alerta(pred)
    }

