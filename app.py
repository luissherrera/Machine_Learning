from flask import Flask, render_template, request
import LinealRegresion
import LinealRegresionNetflix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score
)

app = Flask(__name__)

# --- MAIN ---

@app.route('/')
def home():
    return render_template("home.html")

# --- USE CASES ---

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

# --- LINEAR REGRESSION ---

@app.route('/lr/concepts')
def lr_concepts():
    return render_template("lr_concepts.html")

@app.route('/lr/application', methods=["GET", "POST"])
def lr_application():
    result = None
    if request.method == "POST":
        # get form inputs
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso"])
        visitas = float(request.form["visitas"])
        tiempo = float(request.form["tiempo"])
        compras = float(request.form["compras"])
        descuento = float(request.form["descuento"])

        # make prediction
        result = LinealRegresion.predecir_cliente(
            edad, ingreso, visitas, tiempo, compras, descuento
        )

    return render_template("linealRegresionGrades.html", result=result)

# --- LOGISTIC REGRESSION ---

@app.route('/logistic/concepts')
def logistic_concepts():
    return render_template("logistic_concepts.html")

@app.route('/logistic/application', methods=["GET", "POST"])
def logistic_application():
    prediction = None
    probability = None

    # load and clean dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)

    # encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # select features
    X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
    y = df["Churn"]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred), 3)
    auc = round(roc_auc_score(y_test, y_prob), 3)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("static/roc_curve.png")
    plt.close()

    # prediction from form
    if request.method == "POST":
        tenure = float(request.form["tenure"])
        monthly = float(request.form["MonthlyCharges"])
        total = float(request.form["TotalCharges"])

        data = scaler.transform([[tenure, monthly, total]])

        result = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        prediction = "Churn" if result == 1 else "No Churn"
        probability = round(prob * 100, 2)

    return render_template(
        "logistic_application.html",
        prediction=prediction,
        probability=probability,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc
    )

# --- ASSIGNED MODEL ---

@app.route('/assigned/concepts')
def assigned_concepts():
    return render_template("assigned_concepts.html")

@app.route('/assigned/application')
def assigned_application():
    return render_template("assigned_app.html")

# --- OTHER ---

@app.route('/netflix', methods=["GET", "POST"])
def predictNetflix():
    result = None

    if request.method == "POST":
        descripcion = request.form.get("descripcion")
        result = LinealRegresionNetflix.predecir_genero(descripcion)
    return render_template("linealregresionnetflix.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)