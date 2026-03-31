from flask import Flask, render_template, request

# Importación de módulos de modelos desarrollados previamente
import LinealRegresion
import LinealRegresionNetflix
import NearestCentroidModel

# Inicialización de la aplicación Flask
app = Flask(__name__)

# =========================
# RUTA PRINCIPAL
# =========================
@app.route('/')
def home():
    """
    Renderiza la página principal del sistema.
    """
    return render_template("home.html")


# =========================
# CASOS DE USO
# =========================
@app.route('/case1')
def case1():
    return render_template('case1.html')

@app.route('/case2')
def case2():
    return render_template('case2.html')

@app.route('/case3')
def case3():
    return render_template('case3.html')

@app.route('/case4')
def case4():
    return render_template('case4.html')


# =========================
# REGRESIÓN LINEAL
# =========================
@app.route('/linear-regression/concepts')
def linear_concepts():
    """
    Muestra la sección teórica de regresión lineal.
    """
    return render_template('linear_concepts.html')


@app.route('/linear-regression/application', methods=["GET", "POST"])
def linear_application():
    """
    Permite calcular una predicción de calificación
    en función de las horas de estudio ingresadas por el usuario.
    """
    result = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        result = LinealRegresion.calculateGrade(hours)

    return render_template('linear_application.html', result=result)


# =========================
# REGRESIÓN LOGÍSTICA
# =========================
@app.route('/logistic-regression/concepts')
def logistic_concepts():
    """
    Muestra la sección teórica de regresión logística.
    """
    return render_template('logistic_concepts.html')


@app.route('/logistic-regression/application', methods=["GET", "POST"])
def logistic_application():
    """
    Implementa un modelo de regresión logística para predecir
    la probabilidad de abandono (churn) de un cliente.
    """

    # Importación de librerías necesarias para el modelo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix, accuracy_score,
        precision_score, recall_score,
        f1_score, roc_curve, roc_auc_score
    )

    # Inicialización de variables de salida
    prediction = None
    probability = None

    # -------------------------
    # Carga y preparación de datos
    # -------------------------
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Selección de variables relevantes
    df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]

    # Conversión de datos a formato numérico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)

    # Codificación de la variable objetivo
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Definición de variables independientes y dependiente
    X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y = df['Churn']

    # -------------------------
    # División del conjunto de datos
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Entrenamiento del modelo
    # -------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluación del modelo
    # -------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred), 3)
    auc = round(roc_auc_score(y_test, y_prob), 3)

    # -------------------------
    # Generación de matriz de confusión
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # -------------------------
    # Generación de curva ROC
    # -------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.savefig("static/roc_curve.png")
    plt.close()

    # -------------------------
    # Predicción con datos del usuario
    # -------------------------
    if request.method == "POST":
        try:
            tenure = float(request.form["tenure"])
            monthly = float(request.form["MonthlyCharges"])
            total = float(request.form["TotalCharges"])

            data = [[tenure, monthly, total]]

            pred = model.predict(data)[0]
            prob = model.predict_proba(data)[0][1]

            prediction = "Churn" if pred == 1 else "No Churn"
            probability = round(prob * 100, 2)

        except:
            prediction = "Error"
            probability = 0

    return render_template(
        'logistic_application.html',
        prediction=prediction,
        probability=probability,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc
    )


# =========================
# MODELO ASIGNADO (Nearest Centroid)
# =========================
@app.route('/assigned-model/concepts')
def assigned_concepts():
    return render_template('assigned_concepts.html')


@app.route('/assigned-model/application', methods=["GET", "POST"])
def assigned_application():
    """
    Realiza predicciones utilizando el modelo Nearest Centroid
    y muestra sus métricas de desempeño.
    """
    result = None
    metrics = NearestCentroidModel.get_metrics()

    if request.method == "POST":
        studytime = float(request.form["studytime"])
        failures = float(request.form["failures"])
        absences = float(request.form["absences"])

        result = NearestCentroidModel.predict_student(
            studytime, failures, absences
        )

    return render_template(
        "assigned_application.html",
        result=result,
        metrics=metrics
    )


# =========================
# EJECUCIÓN DE LA APLICACIÓN
# =========================
if __name__ == "__main__":
    app.run(debug=True)