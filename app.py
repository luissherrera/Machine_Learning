from flask import Flask, render_template, request
import LinealRegresion

app = Flask(__name__)

@app.route('/')
def home():
    return "hello flask"

@app.route('/FirstPage')
def firstPage():
    return render_template('index.html')


@app.route('/LinealRegresion', methods=["GET","POST"])
def calculateGrade():

    resultado = None

    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso"])
        visitas = float(request.form["visitas"])
        tiempo = float(request.form["tiempo"])
        compras = float(request.form["compras"])
        descuento = float(request.form["descuento"])

        resultado = LinealRegresion.predecir_cliente(
            edad, ingreso, visitas, tiempo, compras, descuento
        )

    return render_template("linealRegresionGrades.html", result=calculateResults)


if __name__ == "__main__":
    app.run(debug=True)