import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. Cargar datos desde JSON (NDJSON)
df = pd.read_json("gps.json", lines=True)

# 2. Diccionario de límites por ruta y tramo
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

# 3. Agregar columnas de límites y fuera de rango
df["vel_min"] = df.apply(lambda row: limites_ruta.get(row["ruta"], {}).get(row["tramo"], {}).get("min", 0), axis=1)
df["vel_max"] = df.apply(lambda row: limites_ruta.get(row["ruta"], {}).get(row["tramo"], {}).get("max", 999), axis=1)
df["fuera_de_rango"] = df.apply(
    lambda row: 1 if row["velocidad_kmh"] < row["vel_min"] or row["velocidad_kmh"] > row["vel_max"] else 0,
    axis=1
)

# 4. Clasificación del estado de velocidad basada en los límites
def clasificar_alerta(row):
    min_v = row["vel_min"]
    max_v = row["vel_max"]
    if row["velocidad_kmh"] < min_v:
        return 0  # leve
    elif min_v <= row["velocidad_kmh"] < max_v:
        return 1  # grave
    else:  # velocidad_kmh >= max_v
        return 2  # crítica

df["estado_velocidad"] = df.apply(clasificar_alerta, axis=1)

# 5. Codificación de variables categóricas
le_dia = LabelEncoder()
le_clima = LabelEncoder()
le_ruta = LabelEncoder()
le_tramo = LabelEncoder()

df["dia_semana"] = le_dia.fit_transform(df["dia_semana"].astype(str))
df["clima"] = le_clima.fit_transform(df["clima"].astype(str))
df["ruta"] = le_ruta.fit_transform(df["ruta"].astype(str))
df["tramo"] = le_tramo.fit_transform(df["tramo"].astype(str))

# 6. División de datos
X = df.drop(["estado_velocidad", "n_vehiculo", "id", "fecha_hora"], axis=1)
y = df["estado_velocidad"]

# 7. Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# 8. Entrenamiento del modelo
modelo = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
)
modelo.fit(X_bal, y_bal)

# 9. Evaluación
y_pred = modelo.predict(X_bal)
y_proba = modelo.predict_proba(X_bal)

accuracy = accuracy_score(y_bal, y_pred)
conf_matrix = confusion_matrix(y_bal, y_pred)
class_report = classification_report(y_bal, y_pred, output_dict=True)
try:
    auc = roc_auc_score(y_bal, y_proba, multi_class="ovr")
except:
    auc = None

# 10. Guardado del modelo y codificadores
objeto = {
    "modelo": modelo,
    "label_encoders": {
        "dia_semana": le_dia,
        "clima": le_clima,
        "ruta": le_ruta,
        "tramo": le_tramo
    },
    "metricas": {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "roc_auc": auc
    }
}

with open("modelocompleto.lpk", "wb") as f:
    pickle.dump(objeto, f)

print("Modelo entrenado y guardado con éxito usando clasificación 0–1–2 basada en límites por tramo.")