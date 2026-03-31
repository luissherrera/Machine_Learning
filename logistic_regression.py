from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

app = Flask(__name__)

# ==============================
# CARGAR Y PREPARAR DATASET
# ==============================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Limpiar datos
df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Convertir target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODELO
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# MÉTRICAS
# ==============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)
f1 = round(f1_score(y_test, y_pred), 3)
auc = round(roc_auc_score(y_test, y_prob), 3)

# ==============================
# GRÁFICOS
# ==============================

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.title("ROC Curve")
plt.savefig("static/roc_curve.png")
plt.close()

# ==============================
# RUTA PRINCIPAL
# ==============================
@app.route("/logistic", methods=["GET", "POST"])
def logistic():

    prediction = None
    probability = None

    if request.method == "POST":
        try:
            tenure = float(request.form["tenure"])
            monthly = float(request.form["MonthlyCharges"])
            total = float(request.form["TotalCharges"])

            data = np.array([[tenure, monthly, total]])

            pred = model.predict(data)[0]
            prob = model.predict_proba(data)[0][1]

            prediction = "Churn" if pred == 1 else "No Churn"
            probability = round(prob * 100, 2)

        except:
            prediction = "Error"
            probability = 0

    return render_template(
        "logistic_app.html",
        prediction=prediction,
        probability=probability,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc
    )

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)