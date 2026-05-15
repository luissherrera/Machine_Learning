import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)

x = df[["Study Hours"]]
y = df[["Final Grade"]]

model = LinearRegression()
model.fit(x, y)

def calculateGrade(hours):
    hours_df = pd.DataFrame([[hours]], columns=["Study Hours"])
    result = model.predict(hours_df)[0][0]

    # Configurar estilo para que coincida con la UI (modo oscuro)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Fondo consistente con --side-bg
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    
    # Datos originales
    ax.scatter(df["Study Hours"], df["Final Grade"], color='#94a3b8', label='Datos Reales')
    
    # Línea de regresión
    x_range = pd.DataFrame({'Study Hours': range(0, 25)})
    y_pred_range = model.predict(x_range)
    ax.plot(x_range, y_pred_range, color='#38bdf8', linewidth=2, label='Regresión Lineal')
    
    # Punto predecido
    ax.scatter([hours], [result], color='#e50914', s=100, zorder=5, label=f'Predicción ({hours}h, {result:.2f})')
    
    ax.set_title('Study Hours vs Final Grade', color='#f1f5f9')
    ax.set_xlabel('Study Hours', color='#94a3b8')
    ax.set_ylabel('Final Grade', color='#94a3b8')
    
    # Ajustar colores de ejes y bordes
    ax.spines['bottom'].set_color('#334155')
    ax.spines['top'].set_color('#334155') 
    ax.spines['right'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.tick_params(axis='x', colors='#94a3b8')
    ax.tick_params(axis='y', colors='#94a3b8')
    
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f1f5f9')
    ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Guardar a base64
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return result, plot_url