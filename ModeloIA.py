import pandas as pd
import numpy as np
import joblib
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

# 1. Conexion a MongoDB
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

# 3. Clasificar estado de velocidad con nueva regla
def clasificar_estado(v):
    if v >= 90:
        return 3  # critica
    elif v >= 70:
        return 2  # grave
    else:
        return 1  # leve

df["estado_velocidad"] = df["velocidad_kmh"].apply(clasificar_estado)

# 4. Codificación
le_dia = LabelEncoder()
le_clima = LabelEncoder()
le_ruta = LabelEncoder()
le_tramo = LabelEncoder()

df["dia_semana"] = le_dia.fit_transform(df["dia_semana"].astype(str))
df["clima"] = le_clima.fit_transform(df["clima"].astype(str))
df["ruta"] = le_ruta.fit_transform(df["ruta"].astype(str))
df["tramo"] = le_tramo.fit_transform(df["tramo"].astype(str))

# 5. División
X = df.drop(["estado_velocidad", "n_vehiculo"], axis=1)
y = df["estado_velocidad"]

# 6. Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

# 8. Entrenar modelo
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

# 10. Guardado
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

print("Modelo actualizado guardado con niveles: leve <70, grave 70-89, critica >=90")
