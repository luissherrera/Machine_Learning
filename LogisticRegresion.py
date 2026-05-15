import pandas as pd
from sklearn.linear_model import LogisticRegression

# Cargar los datos
df = pd.read_csv('dataset_regresion_logistica.csv')

# Separar características (X) y objetivo (y)
X = df[['edad', 'ingreso_mensual', 'visitas_web_mes', 'tiempo_sitio_min', 'compras_previas', 'descuento_usado']]
y = df['target']

# Inicializar y entrenar el modelo
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

def predict_target(edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado):
    # Crear un DataFrame con los valores de entrada
    input_data = pd.DataFrame([[edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado]], 
                              columns=['edad', 'ingreso_mensual', 'visitas_web_mes', 'tiempo_sitio_min', 'compras_previas', 'descuento_usado'])
    
    # Hacer la predicción
    prediction = model.predict(input_data)[0]
    
    return "Compra" if prediction == 1 else "No Compra"
