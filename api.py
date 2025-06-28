from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

app = FastAPI()

# Cargar modelo y codificadores
with open("modelo_completo.lpk", "rb") as f:
    data = pickle.load(f)

modelo = data["modelo"]
le = data["label_encoders"]
columnas = modelo.feature_names_in_.tolist()

# Mapeo de códigos
alertas = {0: "leve", 1: "grave", 2: "critica"}

class DatosEntrada(BaseModel):
    n_vehiculo: str
    ruta: str
    tramo: str
    velocidad_kmh: float
    temperatura_motor_c: float
    dia_semana: str
    clima: str

@app.post("/alerta")
def predecir_alerta(data: DatosEntrada):
    hora_peru = datetime.now(ZoneInfo("America/Lima")).strftime("%Y-%m-%d %H:%M:%S")

    entrada = data.dict()

    # Codificar variables categóricas
    entrada["dia_semana"] = le["dia_semana"].transform([entrada["dia_semana"]])[0]
    entrada["clima"] = le["clima"].transform([entrada["clima"]])[0]
    entrada["ruta"] = le["ruta"].transform([entrada["ruta"]])[0]
    entrada["tramo"] = le["tramo"].transform([entrada["tramo"]])[0]

    # Preparar entrada para el modelo
    X_nuevo = pd.DataFrame([entrada])[columnas]
    pred = modelo.predict(X_nuevo)[0]

    return {
        "n_vehiculo": data.n_vehiculo,
        "fecha_hora": hora_peru,
        "tipo_alerta": alertas[int(pred)]
    }