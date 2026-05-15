from flask import Flask, render_template, request

# Modelos
import LinealRegresion
import LinealRegresionNetflix
import LogisticRegresion
import NearestCentroidModel
import kmeans_app
import kmeans_manual

app = Flask(__name__)

# =========================
# 🏠 MAIN
# =========================

@app.route('/')
def home():
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
        descripcion = request.form["descripcion"]
        result = LinealRegresionNetflix.predecir_genero(descripcion)

    return render_template('linear_application.html', result=result)


# =========================
# 📉 LOGISTIC REGRESSION
# =========================

@app.route('/logistic-regression/concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')


@app.route('/logistic-regression/application', methods=["GET", "POST"])
def logistic_application():

    result = None

    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso_mensual = float(request.form["ingreso_mensual"])
        visitas_web_mes = float(request.form["visitas_web_mes"])
        tiempo_sitio_min = float(request.form["tiempo_sitio_min"])
        compras_previas = float(request.form["compras_previas"])
        descuento_usado = float(request.form["descuento_usado"])

        result = LogisticRegresion.predict_target(
            edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado
        )

    return render_template('logistic_application.html', result=result)


# =========================
# 🧠 ASSIGNED MODEL (Nearest Centroid)
# =========================

@app.route('/assigned-model/concepts')
def assigned_concepts():
    return render_template('assigned_concepts.html')


@app.route('/assigned-model/application', methods=["GET", "POST"])
def assigned_application():

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
# 🔵 UNSUPERVISED ML – K-MEANS
# =========================

@app.route('/unsupervised/concepts')
def unsupervised_concepts():
    return render_template('unsupervised_concepts.html')


@app.route('/unsupervised/manual-exercise')
def unsupervised_manual():
    data = kmeans_manual.get_manual_data()
    return render_template('unsupervised_manual.html', data=data)


@app.route('/unsupervised/application')
def unsupervised_application():
    results = kmeans_app.run_clustering()
    return render_template('unsupervised_application.html', results=results)


# =========================
# 🚀 RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)
