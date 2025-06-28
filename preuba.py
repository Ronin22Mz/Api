# from fastapi import FastAPI
# from pydantic import BaseModel
# from datetime import datetime
# from zoneinfo import ZoneInfo

# app = FastAPI()

# # Límites por ruta y tramo
# limites_ruta = {
#     "Ruta A": {
#         "Tramo A1": {"min": 40, "max": 60},
#         "Tramo A2": {"min": 45, "max": 65},
#         "Tramo A3": {"min": 50, "max": 70},
#         "Tramo A4": {"min": 40, "max": 55},
#         "Tramo A5": {"min": 45, "max": 60}
#     },
#     "Ruta B": {
#         "Tramo B1": {"min": 50, "max": 75},
#         "Tramo B2": {"min": 45, "max": 70},
#         "Tramo B3": {"min": 55, "max": 80},
#         "Tramo B4": {"min": 50, "max": 70},
#         "Tramo B5": {"min": 40, "max": 60},
#         "Tramo B6": {"min": 45, "max": 65}
#     },
#     "Ruta C": {
#         "Tramo C1": {"min": 40, "max": 55},
#         "Tramo C2": {"min": 45, "max": 65},
#         "Tramo C3": {"min": 50, "max": 70},
#         "Tramo C4": {"min": 55, "max": 75}
#     }
# }

# # Mapeo de códigos de alerta
# alertas = {0: "leve", 1: "grave", 2: "critica"}

# class DatosEntrada(BaseModel):
#     n_vehiculo: str
#     ruta: str
#     tramo: str
#     velocidad_kmh: float
#     temperatura_motor_c: float
#     dia_semana: str
#     clima: str

# @app.post("/alerta")
# def generar_alerta(data: DatosEntrada):
#     hora_peru = datetime.now(ZoneInfo("America/Lima")).strftime("%Y-%m-%d %H:%M:%S")

#     info_tramo = limites_ruta.get(data.ruta, {}).get(data.tramo)

#     if not info_tramo:
#         return {
#             "n_vehiculo": data.n_vehiculo,
#             "fecha_hora": hora_peru,
#             "codigo_alerta": -1,
#             "tipo_alerta": "ruta o tramo no definidos"
#         }

#     min_vel = info_tramo["min"]
#     max_vel = info_tramo["max"]
#     velocidad = data.velocidad_kmh

#     if velocidad < min_vel:
#         codigo_alerta = 0  # leve
#     elif min_vel <= velocidad <= max_vel:
#         codigo_alerta = 1  # grave
#     else:
#         codigo_alerta = 2  # critica

#     return {
#         "n_vehiculo": data.n_vehiculo,
#         "fecha_hora": hora_peru,
#         "tipo_alerta": alertas[codigo_alerta]
#     }