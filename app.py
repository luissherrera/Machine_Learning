from flask import Flask, render_template, request
import LinealRegresion
import LinealRegresionNetflix
# Importar aquí tu futura lógica de Logística cuando esté lista
# import LogisticRegresion 

app = Flask(__name__)

# --- MAIN ROUTES ---

@app.route('/')
def home():
    return render_template("home.html")

# --- ML USE CASES (1-4) ---

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

# --- SUPERVISED MACHINE LEARNING SECTIONS ---

# 1. Linear Regression
@app.route('/lr/concepts')
def lr_concepts():
    return render_template("lr_concepts.html")

@app.route('/lr/application', methods=["GET", "POST"])
def lr_application():
    resultado = None
    if request.method == "POST":
        # Extraer datos del formulario (asegúrate de que los 'name' en HTML coincidan)
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso"])
        visitas = float(request.form["visitas"])
        tiempo = float(request.form["tiempo"])
        compras = float(request.form["compras"])
        descuento = float(request.form["descuento"])

        resultado = LinealRegresion.predecir_cliente(
            edad, ingreso, visitas, tiempo, compras, descuento
        )
    # Cambié el nombre del template a uno más descriptivo si deseas renombrarlo
    return render_template("linealRegresionGrades.html", result=resultado)

# 2. Logistic Regression
@app.route('/logistic/concepts')
def logistic_concepts():
    return render_template("logistic_concepts.html")

@app.route('/logistic/application', methods=["GET", "POST"])
def logistic_application():
    resultado = None
    # Aquí irá la lógica de predicción de tu CSV de logística
    return render_template("logistic_app.html", result=resultado)

# 3. Assigned Model (Example: KNN or SVM)
@app.route('/assigned/concepts')
def assigned_concepts():
    return render_template("assigned_concepts.html")

@app.route('/assigned/application')
def assigned_application():
    return render_template("assigned_app.html")

# --- OTHER ROUTES ---

@app.route('/netflix', methods=["GET", "POST"])
def predictNetflix():
    resultado = None
    if request.method == "POST":
        descripcion = request.form.get("descripcion")
        resultado = LinealRegresionNetflix.predecir_genero(descripcion)
    return render_template("linealregresionnetflix.html", result=resultado)

if __name__ == "__main__":
    app.run(debug=True)