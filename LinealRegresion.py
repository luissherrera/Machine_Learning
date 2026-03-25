import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar datos (CORREGIDO: usa coma)
df = pd.read_csv("dataset_regresion_logistica.csv")

# 2. Verificar columnas
print("Columnas del dataset:")
print(df.columns)

# 3. Ver primeras filas
print("\nPrimeros datos:")
print(df.head())

# 4. Renombrar columnas (opcional)
df = df.rename(columns={
    "ingreso_men": "ingreso_mensual",
    "visitas_web": "visitas_web_mes",
    "tiempo_sitio": "tiempo_sitio_min",
    "compras_pre": "compras_previas",
    "descuento_u": "descuento_usado"
})

# 5. Variables independientes (X) y dependiente (y)
X = df[[
    "edad",
    "ingreso_mensual",
    "visitas_web_mes",
    "tiempo_sitio_min",
    "compras_previas",
    "descuento_usado"
]]

y = df["target"]

# 6. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Predicciones
y_pred = model.predict(X_test)

# 9. Evaluación
print("\nExactitud (Accuracy):", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

# 10. Función para predecir nuevos datos
def predecir_cliente(edad, ingreso, visitas, tiempo, compras, descuento):
    nuevo = pd.DataFrame([{
        "edad": edad,
        "ingreso_mensual": ingreso,
        "visitas_web_mes": visitas,
        "tiempo_sitio_min": tiempo,
        "compras_previas": compras,
        "descuento_usado": descuento
    }])
    
    prediccion = model.predict(nuevo)[0]
    
    return "Compra" if prediccion == 1 else "No compra"

# 11. Ejemplo
resultado = predecir_cliente(30, 3000000, 10, 8.5, 2, 1)
print("\nPredicción para nuevo cliente:", resultado)