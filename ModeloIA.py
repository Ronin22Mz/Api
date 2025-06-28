import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. Conexión a MongoDB
uri = "mongodb+srv://benja15mz:123@database.5iimvyd.mongodb.net/"
client = MongoClient(uri)
db = client["GPS"]
col = db["Gps"]

# 2. Cargar datos
projection = {
    "_id": 0,
    "n_vehiculo": 1,
    "ruta": 1,
    "tramo": 1,
    "velocidad_kmh": 1,
    "temperatura_motor_c": 1,
    "dia_semana": 1,
    "clima": 1
}
df = pd.DataFrame(list(col.find({}, projection)))

# 3. Clasificación del estado de velocidad según norma legal
def clasificar_estado(v):
    if v <= 50:
        return 0  # leve
    elif 51 <= v <= 65:
        return 1  # grave
    else:
        return 2  # crítica

df["estado_velocidad"] = df["velocidad_kmh"].apply(clasificar_estado)

# 4. Codificación de variables categóricas
le_dia = LabelEncoder()
le_clima = LabelEncoder()
le_ruta = LabelEncoder()
le_tramo = LabelEncoder()

df["dia_semana"] = le_dia.fit_transform(df["dia_semana"].astype(str))
df["clima"] = le_clima.fit_transform(df["clima"].astype(str))
df["ruta"] = le_ruta.fit_transform(df["ruta"].astype(str))
df["tramo"] = le_tramo.fit_transform(df["tramo"].astype(str))

# 5. División de datos
X = df.drop(["estado_velocidad", "n_vehiculo"], axis=1)
y = df["estado_velocidad"]

# 6. Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# 7. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

# 8. Entrenamiento del modelo
modelo = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
)
modelo.fit(X_train, y_train)

# 9. Evaluación
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
try:
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
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

with open("modelo_completo.lpk", "wb") as f:
    pickle.dump(objeto, f)

print("Modelo actualizado guardado")