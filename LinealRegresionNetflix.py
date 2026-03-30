import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Cargar datos
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Netflix_data.csv")
df = pd.read_csv(csv_path)

# 2. Preprocesar datos
df = df.dropna(subset=['Description', 'Type']).copy()

# Tomar solo el primer género como categoría principal para simplificar
df['Primary_Genre'] = df['Type'].apply(lambda x: x.split(',')[0].strip())

X = df['Description']
y = df['Primary_Genre']

# 3. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear pipeline con TF-IDF y Regresión Logística (Clasificación de Texto)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Entrenar el modelo
model.fit(X_train, y_train)

# 5. Función de predicción a ser importada por app.py
def predecir_genero(descripcion):
    prediccion = model.predict([descripcion])[0]
    return f"{prediccion}"
