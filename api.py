from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from datetime import datetime

app = FastAPI()

# Cargar modelo y codificadores una sola vez
with open("modelo_completo.lpk", "rb") as f:
    data = pickle.load(f)

modelo = data["modelo"]
le = data["label_encoders"]
columnas = modelo.feature_names_in_.tolist()

# Traducir código de predicción a tipo de alerta
def interpretar_alerta(codigo):
    return {1: "leve", 2: "grave", 3: "critica"}.get(codigo, "desconocida")

# Esquema de entrada para la API
class DatosEntrada(BaseModel):
    n_vehiculo: str
    ruta: str
    tramo: str
    velocidad_kmh: float
    temperatura_motor_c: float
    dia_semana: str
    clima: str

# Endpoint principal
@app.post("/alerta")
def predecir_alerta(data: DatosEntrada):
    entrada = data.dict()

    # Codificación de variables categóricas
    entrada["dia_semana"] = le["dia_semana"].transform([entrada["dia_semana"]])[0]
    entrada["clima"] = le["clima"].transform([entrada["clima"]])[0]
    entrada["ruta"] = le["ruta"].transform([entrada["ruta"]])[0]
    entrada["tramo"] = le["tramo"].transform([entrada["tramo"]])[0]

    # Preparar entrada y predecir
    X_nuevo = pd.DataFrame([entrada])[columnas]
    pred = modelo.predict(X_nuevo)[0]

    return {
        "n_vehiculo": data.n_vehiculo,
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "estado_velocidad": int(pred),
        "tipo_alerta": interpretar_alerta(pred)
    }