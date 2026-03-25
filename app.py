from flask import Flask, render_template, request
import LinealRegresion

app = Flask(__name__)

# -------------------------------
# Ruta principal: HOME
# -------------------------------
@app.route('/')
@app.route('/home')
def home():
    return "hello flask"

# Página opcional
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

    return render_template("linealRegresionGrades.html", result=resultado)

# -------------------------------
# Ejecutar app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)