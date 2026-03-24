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

    calculateResults = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResults = LinealRegresion.calculateGrade(hours)

    return render_template("linealRegresionGrades.html", result=calculateResults)

@app.route('/casosdeuso')
def casosdeuso():
    return render_template('casosdeuso1.html')

if __name__ == "__main__":
    app.run(debug=True)