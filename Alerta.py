from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

app = FastAPI()

# Cargar modelo y codificadores
try:
    with open("modelo_completo.lpk", "rb") as f:
        data = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

modelo = data["modelo"]
le = data["label_encoders"]
columnas = modelo.feature_names_in_.tolist()

# Diccionario de límites por ruta y tramo
limites_ruta = {
    "Ruta A": {
        "Tramo A1": {"min": 40, "max": 60},
        "Tramo A2": {"min": 45, "max": 65},
        "Tramo A3": {"min": 50, "max": 70},
        "Tramo A4": {"min": 40, "max": 55},
        "Tramo A5": {"min": 45, "max": 60}
    },
    "Ruta B": {
        "Tramo B1": {"min": 50, "max": 75},
        "Tramo B2": {"min": 45, "max": 70},
        "Tramo B3": {"min": 55, "max": 80},
        "Tramo B4": {"min": 50, "max": 70},
        "Tramo B5": {"min": 40, "max": 60},
        "Tramo B6": {"min": 45, "max": 65}
    },
    "Ruta C": {
        "Tramo C1": {"min": 40, "max": 55},
        "Tramo C2": {"min": 45, "max": 65},
        "Tramo C3": {"min": 50, "max": 70},
        "Tramo C4": {"min": 55, "max": 75}
    }
}

# Mapeo de códigos de alerta
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

    # Validar ruta y tramo
    info_tramo = limites_ruta.get(data.ruta, {}).get(data.tramo)
    if not info_tramo:
        return {
            "n_vehiculo": data.n_vehiculo,
            "fecha_hora": hora_peru,
            "tipo_alerta": "ruta o tramo no definidos"
        }

    # Calcular campos derivados
    entrada["vel_min"] = info_tramo["min"]
    entrada["vel_max"] = info_tramo["max"]
    entrada["fuera_de_rango"] = 1 if data.velocidad_kmh < info_tramo["min"] or data.velocidad_kmh > info_tramo["max"] else 0
    entrada["km_por_tramo_minimo"] = info_tramo["min"]
    entrada["km_por_tramo_maximo"] = info_tramo["max"]

    # Codificar variables categóricas
    try:
        entrada["dia_semana"] = le["dia_semana"].transform([entrada["dia_semana"]])[0]
        entrada["clima"] = le["clima"].transform([entrada["clima"]])[0]
        entrada["ruta"] = le["ruta"].transform([entrada["ruta"]])[0]
        entrada["tramo"] = le["tramo"].transform([entrada["tramo"]])[0]
    except ValueError as e:
        return {
            "n_vehiculo": data.n_vehiculo,
            "fecha_hora": hora_peru,
            "tipo_alerta": f"valor no reconocido: {e}"
        }

    # Filtrar solo las columnas que el modelo espera
    entrada = {k: v for k, v in entrada.items() if k in columnas}

    # Hacer predicción
    try:
        X_nuevo = pd.DataFrame([entrada])[columnas]
        pred = modelo.predict(X_nuevo)[0]
    except Exception as e:
        return {
            "n_vehiculo": data.n_vehiculo,
            "fecha_hora": hora_peru,
            "tipo_alerta": f"error en predicción: {e}"
        }

    return {
        "n_vehiculo": data.n_vehiculo,
        "fecha_hora": hora_peru,
        "tipo_alerta": alertas[int(pred)]
    }