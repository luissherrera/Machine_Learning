from flask import Flask, render_template, request

# Modelos
import LinealRegresion
import LinealRegresionNetflix
# import LogisticRegresion
import NearestCentroidModel

# Inicialización de la aplicación Flask
app = Flask(__name__)

# =========================
# 🏠 MAIN
# =========================

@app.route('/')
def home():
    """
    Renderiza la página principal del sistema.
    """
    return render_template("home.html")


# =========================
# 📊 USE CASES
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
# 📈 LINEAR REGRESSION
# =========================

@app.route('/linear-regression/concepts')
def linear_concepts():
    return render_template('linear_concepts.html')


@app.route('/linear-regression/application', methods=["GET", "POST"])
def linear_application():

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

# =========================
# 📉 LOGISTIC REGRESSION
# =========================

@app.route('/logistic-regression/concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')

@app.route('/logistic/application', methods=["GET", "POST"])
def logistic_application():

    result = None

    if request.method == "POST":
        # Aquí luego conectas tu modelo
        pass

    return render_template('logistic_application.html', result=result)


# =========================
# 🧠 ASSIGNED MODEL (Nearest Centroid)
# =========================

@app.route('/assigned-model/concepts')
def assigned_concepts():
    return render_template("assigned_concepts.html")

@app.route('/assigned/application')
def assigned_application():

    result = None
    metrics = NearestCentroidModel.get_metrics()  # 🔥 NUEVO

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
        metrics=metrics  # 🔥 NUEVO
    )

# =========================
# 🚀 RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)