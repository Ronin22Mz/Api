from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

app = FastAPI()

# Cargar modelo y codificadores
with open("modelo_completo.lpk", "rb") as f:
    data = pickle.load(f)

modelo = data["modelo"]
le = data["label_encoders"]
columnas = modelo.feature_names_in_.tolist()

# Límites por ruta y tramo
limites_ruta = {
    "Ruta A": {
        "Tramo A1": {"min": 40, "max": 60, "temp": 85},
        "Tramo A2": {"min": 45, "max": 65, "temp": 88},
        "Tramo A3": {"min": 50, "max": 70, "temp": 90},
        "Tramo A4": {"min": 40, "max": 55, "temp": 82},
        "Tramo A5": {"min": 45, "max": 60, "temp": 87}
    },
    "Ruta B": {
        "Tramo B1": {"min": 50, "max": 75, "temp": 92},
        "Tramo B2": {"min": 45, "max": 70, "temp": 88},
        "Tramo B3": {"min": 55, "max": 80, "temp": 95},
        "Tramo B4": {"min": 50, "max": 70, "temp": 90},
        "Tramo B5": {"min": 40, "max": 60, "temp": 85},
        "Tramo B6": {"min": 45, "max": 65, "temp": 87}
    },
    "Ruta C": {
        "Tramo C1": {"min": 40, "max": 55, "temp": 83},
        "Tramo C2": {"min": 45, "max": 65, "temp": 86},
        "Tramo C3": {"min": 50, "max": 70, "temp": 89},
        "Tramo C4": {"min": 55, "max": 75, "temp": 93}
    }
}

def interpretar_alerta(codigo):
    return {0: "leve", 1: "grave", 2: "critica"}.get(codigo, "desconocida")

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
    entrada = data.dict()

    # Codificación de variables categóricas
    entrada["dia_semana"] = le["dia_semana"].transform([entrada["dia_semana"]])[0]
    entrada["clima"] = le["clima"].transform([entrada["clima"]])[0]
    entrada["ruta"] = le["ruta"].transform([entrada["ruta"]])[0]
    entrada["tramo"] = le["tramo"].transform([entrada["tramo"]])[0]

    # Preparar entrada y predecir
    X_nuevo = pd.DataFrame([entrada])[columnas]
    pred = modelo.predict(X_nuevo)[0]

    # Hora local de Perú
    hora_peru = datetime.now(ZoneInfo("America/Lima")).strftime("%Y-%m-%d %H:%M:%S")

    # Validación de velocidad y temperatura
    ruta = data.ruta
    tramo = data.tramo
    velocidad = data.velocidad_kmh
    temperatura = data.temperatura_motor_c

    info_tramo = limites_ruta.get(ruta, {}).get(tramo)
    if info_tramo:
        min_vel = info_tramo["min"]
        max_vel = info_tramo["max"]
        temp_esperada = info_tramo["temp"]

        if velocidad < min_vel:
            estado_vel = "por debajo del mínimo"
        elif velocidad > max_vel:
            estado_vel = "por encima del máximo"
        else:
            estado_vel = "dentro del rango"

        estado_temp = (
            "normal" if abs(temperatura - temp_esperada) <= 5
            else "anómala"
        )
    else:
        estado_vel = "tramo o ruta no definidos"
        estado_temp = "tramo o ruta no definidos"

    return {
        "n_vehiculo": data.n_vehiculo,
        "fecha_hora": hora_peru,
        "estado_velocidad": int(pred),
        "tipo_alerta": interpretar_alerta(pred),
    }