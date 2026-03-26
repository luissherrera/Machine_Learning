import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(filepath):
    # 1. Cargar el dataset
    print(f"Cargando datos desde {filepath}...")
    df = pd.read_csv(filepath)
    
    # 2. Filtrar solo Películas ('Movie')
    # Predecir la duración en minutos es más claro que mezclar con 'Seasons' (temporadas) de TV Shows
    df_movies = df[df['Category'] == 'Movie'].copy()
    
    # 3. Limpiar la columna 'Duration' para extraer los minutos como número
    # Ej: '93 min' -> 93.0
    df_movies['Duration_num'] = df_movies['Duration'].str.replace(' min', '').astype(float)
    
    # 4. Limpiar 'Release_Date' para extraer el año de lanzamiento
    df_movies['Release_Year'] = pd.to_datetime(df_movies['Release_Date'], errors='coerce').dt.year
    
    # 5. Seleccionar características (Features) y variable objetivo (Target)
    # Utilizaremos 'Rating', el año de lanzamiento y el 'Type' (Géneros)
    features = ['Rating', 'Release_Year', 'Type']
    target = 'Duration_num'
    
    data = df_movies[features + [target]].dropna()
    
    X = data[features]
    y = data[target]
    
    return X, y

def build_model():
    # Preprocesamiento para datos categóricos
    categorical_features = ['Rating', 'Type']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Preprocesamiento para datos numéricos
    numeric_features = ['Release_Year']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Combinar preprocesadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Crear el pipeline con un regresor lineal
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])
    return model

def main():
    filepath = 'Netflix_data.csv'
    X, y = load_and_preprocess_data(filepath)
    
    print(f"Forma del dataset después del preprocesamiento: {X.shape}")
    
    # Dividir los datos en conjunto de entrenamiento y prueba (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construir y entrenar el modelo
    model = build_model()
    print("Entrenando el modelo...")
    model.fit(X_train, y_train)
    
    # Predecir y evaluar
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Evaluación del Modelo ---")
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Score R^2: {r2:.4f}")
    
    # Realizar algunas predicciones de ejemplo
    print("\n--- Ejemplos de Predicción ---")
    example_data = X_test.iloc[:5]
    example_preds = model.predict(example_data)
    example_actuals = y_test.iloc[:5].values
    
    for i in range(5):
        print(f"Características: {example_data.iloc[i].to_dict()}")
        print(f"Duración Predicha: {example_preds[i]:.0f} min | Duración Real: {example_actuals[i]} min\n")

if __name__ == '__main__':
    main()
