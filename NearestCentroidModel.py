import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid

# 🔥 NUEVO (IMPORTS PARA MÉTRICAS Y GRÁFICA)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 🔥 CARGAR DATASET
df = pd.read_csv("dataset/student-mat.csv")

# 🔥 LIMPIAR COLUMNAS
df.columns = df.columns.str.strip()

print("Columnas:", df.columns)

# 🔥 VARIABLES
X = df[["studytime", "failures", "absences"]]

# 🔥 TARGET
df["target"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
y = df["target"]

# 🔥 DIVISIÓN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 MODELO
model = NearestCentroid()
model.fit(X_train, y_train)

# =========================
# 🔥 NUEVO: EVALUACIÓN REAL
# =========================

y_pred = model.predict(X_test)

# 🔥 MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_test, y_pred)

# 🔥 GRAFICAR MATRIZ
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# 🔥 GUARDAR IMAGEN
plt.savefig("static/confusion_matrix.png")
plt.close()

# 🔥 MÉTRICAS
accuracy = round(accuracy_score(y_test, y_pred), 2)
precision = round(precision_score(y_test, y_pred), 2)
recall = round(recall_score(y_test, y_pred), 2)
f1 = round(f1_score(y_test, y_pred), 2)

# 🔥 FUNCIÓN PARA FLASK (PREDICCIÓN)
def predict_student(studytime, failures, absences):
    data = [[studytime, failures, absences]]
    prediction = model.predict(data)
    return int(prediction[0])

# 🔥 FUNCIÓN PARA MÉTRICAS
def get_metrics():
    return accuracy, precision, recall, f1