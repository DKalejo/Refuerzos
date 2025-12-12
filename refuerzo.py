# --------------------------------------------------------------
# 1. Importación de librerías necesarias
# --------------------------------------------------------------

import pandas as pd                          # Manipulación de datos
import numpy as np                           # Operaciones numéricas
from sklearn.model_selection import train_test_split  # Dividir datos en train/test
from sklearn.preprocessing import StandardScaler       # Normalización de variables
from sklearn.ensemble import RandomForestClassifier    # Modelo de clasificación
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# Métricas de evaluación


# --------------------------------------------------------------
# 2. Cargar el dataset desde GitHub
# --------------------------------------------------------------

# URL del archivo CSV con información histórica de mundiales
url = "https://raw.githubusercontent.com/aperezn298/CienciaDatosSENA/main/04Datasets/world_cup_prediction_dataset.xlsx"

df = pd.read_excel(url)  # Cargar el dataset en un DataFrame
print(df.head())       # Mostrar las primeras filas para entender la estructura


# --------------------------------------------------------------
# 3. Exploración y comprensión del dataset
# --------------------------------------------------------------

print(df.info())             # Revisar tipos de datos y columnas
print(df.describe())         # Estadísticas básicas de las variables numéricas
print(df.isnull().sum())     # Contar valores faltantes en cada columna


# --------------------------------------------------------------
# 4. Preparación y limpieza de datos
# --------------------------------------------------------------

# Guardamos los nombres de los equipos para el ranking final
teams = df["Team"]

df = df.drop(columns=["Team"])

# Llenamos los valores nulos con la media de cada columna
df = df.fillna(df.mean())

# Variable objetivo (lo que queremos predecir)
y = df["Champion"]

# Variables predictoras (todas menos la etiqueta y otras irrelevantes)
X = df.drop(columns=["Champion", "Year"])  
# "Year" se elimina porque no influye directamente en si un equipo es campeón


# --------------------------------------------------------------
# 5. Normalización de variables numéricas
# --------------------------------------------------------------

scaler = StandardScaler()        # Crear el objeto para normalizar
X_scaled = scaler.fit_transform(X)  # Ajustar (fit) y transformar (transform)


# --------------------------------------------------------------
# 6. División de los datos en entrenamiento y prueba
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,    # Datos ya normalizados
    y,           # Etiquetas
    test_size=0.30,     # 30% de los datos para prueba
    random_state=42,    # Semilla para reproducibilidad
    stratify=y          # Asegurar proporción de clases en train/test
)


# --------------------------------------------------------------
# 7. Entrenamiento del modelo de Machine Learning
# --------------------------------------------------------------

# Se selecciona Random Forest porque:
# - Maneja bien datos tabulares
# - Es robusto al sobreajuste
# - Maneja relaciones no lineales
model = RandomForestClassifier(
    n_estimators=200,  # Número de árboles del bosque
    max_depth=5,       # Reducimos la profundidad para evitar sobreajuste
    random_state=42,
    class_weight='balanced'  # Dar más peso a la clase minoritaria (campeones)
)

model.fit(X_train, y_train)   # Entrenar el modelo con los datos


# --------------------------------------------------------------
# 8. Evaluación del modelo con métricas
# --------------------------------------------------------------

# Predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Cálculo de métricas de evaluación
acc = accuracy_score(y_test, y_pred)       # Exactitud
prec = precision_score(y_test, y_pred)     # Precisión
rec = recall_score(y_test, y_pred)         # Sensibilidad
cm = confusion_matrix(y_test, y_pred)      # Matriz de confusión

# Mostrar métricas
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("Matriz de Confusión:\n", cm)


# --------------------------------------------------------------
# 9. Generación de probabilidades de ser campeón
# --------------------------------------------------------------

# Usamos predict_proba() para obtener probabilidades en lugar de una clase
probabilities = model.predict_proba(X_scaled)[:, 1]  # Probabilidad de ser campeón (clase 1)

ranking_df = pd.DataFrame({
    "Team": teams,
    "Probability_Champion": probabilities
})

ranking = ranking_df.sort_values(by="Probability_Champion", ascending=False).reset_index(drop=True)

# Mostrar las 10 selecciones con mayor probabilidad
print("TOP 10 selecciones con mayor probabilidad:")
print(ranking.head(10))
