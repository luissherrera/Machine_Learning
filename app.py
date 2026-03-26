from flask import Flask, render_template, request
import LinealRegresion

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

# Página opcional
@app.route('/FirstPage')
def firstPage():
    return render_template('index.html')

# -------------------------------
# Página de Casos de Uso (Dashboard)
# -------------------------------
@app.route('/use_cases')
def use_cases():
    return render_template("use_cases.html")

# -------------------------------
# Rutas de los casos individuales
# -------------------------------
@app.route('/caso1')
def caso1():
    return render_template('caso1.html')

@app.route('/caso2')
def caso2():
    return render_template('caso2.html')

@app.route('/caso3')
def caso3():
    return render_template('caso3.html')

@app.route('/caso4')
def caso4():
    return render_template('caso4.html')

# -------------------------------
# Ruta de predicción LinealRegresion
# -------------------------------
@app.route('/LinealRegresion', methods=["GET", "POST"])
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

# -------------------------------
# Ejecutar app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)