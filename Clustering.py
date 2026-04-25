from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def getDataset():
    return [
        {"nombre": "Ana", "edad": 22, "ingresos": 1200, "gasto": 300},
        {"nombre": "Luis", "edad": 25, "ingresos": 1500, "gasto": 350},
        {"nombre": "Carlos", "edad": 23, "ingresos": 1300, "gasto": 280},
        {"nombre": "Marta", "edad": 45, "ingresos": 4000, "gasto": 1200},
        {"nombre": "Sofía", "edad": 50, "ingresos": 4200, "gasto": 1400},
        {"nombre": "Jorge", "edad": 47, "ingresos": 3900, "gasto": 1100},
        {"nombre": "Elena", "edad": 31, "ingresos": 2500, "gasto": 700},
        {"nombre": "Pedro", "edad": 33, "ingresos": 2700, "gasto": 750},
        {"nombre": "Laura", "edad": 29, "ingresos": 2400, "gasto": 680},
        {"nombre": "Andrés", "edad": 52, "ingresos": 5000, "gasto": 1600},
        {"nombre": "Camila", "edad": 21, "ingresos": 1100, "gasto": 250},
        {"nombre": "Diego", "edad": 38, "ingresos": 3200, "gasto": 900},
        {"nombre": "Ana", "edad": 22, "ingresos": 1200, "gasto": 300},
        {"nombre": "Luis", "edad": 25, "ingresos": 1500, "gasto": 350},
        {"nombre": "Carlos", "edad": 23, "ingresos": 1300, "gasto": 280},
        {"nombre": "Maria", "edad": 30, "ingresos": 2000, "gasto": 500},
        {"nombre": "Pedro", "edad": 28, "ingresos": 1800, "gasto": 450},
        {"nombre": "Sofia", "edad": 21, "ingresos": 1100, "gasto": 250},
        {"nombre": "Jorge", "edad": 35, "ingresos": 2500, "gasto": 700},
        {"nombre": "Lucia", "edad": 27, "ingresos": 1700, "gasto": 400},
        {"nombre": "Diego", "edad": 24, "ingresos": 1400, "gasto": 320},
        {"nombre": "Valentina", "edad": 29, "ingresos": 1900, "gasto": 480},
        {"nombre": "Andres", "edad": 32, "ingresos": 2200, "gasto": 600},
        {"nombre": "Camila", "edad": 26, "ingresos": 1600, "gasto": 370},
        {"nombre": "Daniel", "edad": 31, "ingresos": 2100, "gasto": 550},
        {"nombre": "Paula", "edad": 23, "ingresos": 1350, "gasto": 290},
        {"nombre": "Miguel", "edad": 34, "ingresos": 2400, "gasto": 680},
        {"nombre": "Laura", "edad": 28, "ingresos": 1750, "gasto": 420},
        {"nombre": "Fernando", "edad": 36, "ingresos": 2600, "gasto": 750},
        {"nombre": "Natalia", "edad": 27, "ingresos": 1650, "gasto": 390},
        {"nombre": "Sebastian", "edad": 33, "ingresos": 2300, "gasto": 640},
        {"nombre": "Daniela", "edad": 24, "ingresos": 1450, "gasto": 330},
        {"nombre": "Ricardo", "edad": 38, "ingresos": 2800, "gasto": 800},
        {"nombre": "Sara", "edad": 22, "ingresos": 1250, "gasto": 310},
        {"nombre": "Hector", "edad": 40, "ingresos": 3000, "gasto": 900},
        {"nombre": "Angela", "edad": 29, "ingresos": 1850, "gasto": 460},
        {"nombre": "Ivan", "edad": 35, "ingresos": 2550, "gasto": 720},
        {"nombre": "Claudia", "edad": 31, "ingresos": 2050, "gasto": 540},
        {"nombre": "Oscar", "edad": 37, "ingresos": 2700, "gasto": 780},
        {"nombre": "Patricia", "edad": 28, "ingresos": 1780, "gasto": 430},
        {"nombre": "Julian", "edad": 26, "ingresos": 1550, "gasto": 360},
        {"nombre": "Carolina", "edad": 34, "ingresos": 2400, "gasto": 660},
        {"nombre": "Esteban", "edad": 30, "ingresos": 2000, "gasto": 520},
        {"nombre": "Tatiana", "edad": 25, "ingresos": 1500, "gasto": 340},
        {"nombre": "Gustavo", "edad": 39, "ingresos": 2900, "gasto": 850},
        {"nombre": "Monica", "edad": 27, "ingresos": 1680, "gasto": 395},
        {"nombre": "Alberto", "edad": 41, "ingresos": 3100, "gasto": 920},
        {"nombre": "Veronica", "edad": 33, "ingresos": 2250, "gasto": 610},
        {"nombre": "Rafael", "edad": 36, "ingresos": 2600, "gasto": 740},
        {"nombre": "Diana", "edad": 24, "ingresos": 1420, "gasto": 315},
        {"nombre": "Mauricio", "edad": 38, "ingresos": 2750, "gasto": 790},
        {"nombre": "Alejandra", "edad": 29, "ingresos": 1880, "gasto": 470},
        {"nombre": "Cristian", "edad": 32, "ingresos": 2150, "gasto": 580},
        {"nombre": "Liliana", "edad": 26, "ingresos": 1580, "gasto": 365},
        {"nombre": "Fabian", "edad": 35, "ingresos": 2500, "gasto": 710},
        {"nombre": "Marcela", "edad": 28, "ingresos": 1800, "gasto": 440},
        {"nombre": "Kevin", "edad": 23, "ingresos": 1320, "gasto": 285},
        {"nombre": "Adriana", "edad": 31, "ingresos": 2080, "gasto": 530},
        {"nombre": "Javier", "edad": 37, "ingresos": 2680, "gasto": 770},
        {"nombre": "Paola", "edad": 27, "ingresos": 1700, "gasto": 405},
        {"nombre": "Brayan", "edad": 22, "ingresos": 1280, "gasto": 300},
        {"nombre": "Yuli", "edad": 25, "ingresos": 1480, "gasto": 345},
        {"nombre": "Nicolas", "edad": 30, "ingresos": 1980, "gasto": 510},
        {"nombre": "Karen", "edad": 28, "ingresos": 1820, "gasto": 435},
        {"nombre": "Santiago", "edad": 34, "ingresos": 2380, "gasto": 670},
        {"nombre": "Daniel", "edad": 33, "ingresos": 2280, "gasto": 620},
        {"nombre": "Andrea", "edad": 26, "ingresos": 1600, "gasto": 375},
        {"nombre": "Felipe", "edad": 35, "ingresos": 2520, "gasto": 705},
        {"nombre": "Lorena", "edad": 29, "ingresos": 1900, "gasto": 465},
        {"nombre": "Samuel", "edad": 24, "ingresos": 1380, "gasto": 310},
        {"nombre": "Vanessa", "edad": 32, "ingresos": 2200, "gasto": 590},
        {"nombre": "Cesar", "edad": 36, "ingresos": 2580, "gasto": 730},
        {"nombre": "Melissa", "edad": 27, "ingresos": 1720, "gasto": 400},
        {"nombre": "David", "edad": 31, "ingresos": 2100, "gasto": 560},
        {"nombre": "Jessica", "edad": 23, "ingresos": 1340, "gasto": 295},
        {"nombre": "Leonardo", "edad": 38, "ingresos": 2800, "gasto": 810},
        {"nombre": "Paula", "edad": 28, "ingresos": 1780, "gasto": 420},
        {"nombre": "Edwin", "edad": 40, "ingresos": 3000, "gasto": 880},
        {"nombre": "Tatiana", "edad": 26, "ingresos": 1620, "gasto": 370},
        {"nombre": "Julio", "edad": 35, "ingresos": 2480, "gasto": 700},
        {"nombre": "Clara", "edad": 29, "ingresos": 1850, "gasto": 455},
        {"nombre": "Henry", "edad": 37, "ingresos": 2700, "gasto": 760},
        {"nombre": "Rosa", "edad": 30, "ingresos": 2000, "gasto": 520},
        {"nombre": "Oscar", "edad": 34, "ingresos": 2350, "gasto": 680},
        {"nombre": "Elena", "edad": 27, "ingresos": 1680, "gasto": 395},
        {"nombre": "Victor", "edad": 39, "ingresos": 2900, "gasto": 840},
        {"nombre": "Gloria", "edad": 31, "ingresos": 2100, "gasto": 560},
        {"nombre": "Raul", "edad": 36, "ingresos": 2600, "gasto": 740},
        {"nombre": "Beatriz", "edad": 28, "ingresos": 1750, "gasto": 430}
    ]

def ApplyClusteringkmeans():
    data = getDataset()

    x = [[person["edad"], person["ingresos"], person["gasto"]] for person in data]

    scaler = StandardScaler()
    XScaled = scaler.fit_transform(x)
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(XScaled)

    result = []

    for i, person in enumerate(data):
        row = person.copy()
        row["Cluster"] = int(labels[i])
        result.append(row)

    summaryClusters = {}

    for label in labels:
        label = int(label)
        summaryClusters[label] = summaryClusters.get(label, 0) + 1

    centers = model.cluster_centers_.tolist()

    # 🔥 GENERATE CLUSTERING PLOT
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    img_path = os.path.join(static_dir, "clustering_plot.png")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=[p["ingresos"] for p in result],
        y=[p["gasto"] for p in result],
        hue=[p["Cluster"] for p in result],
        palette='viridis',
        s=150,
        edgecolor='black',
        alpha=0.8
    )
    plt.xlabel("Ingresos (Income)")
    plt.ylabel("Gasto (Spend)")
    plt.title("K-Means Clustering: Income vs Spend")
    plt.legend(title="Cluster")
    plt.savefig(img_path)
    plt.close()

    return {
        "results": result,
        "summaryClusters": summaryClusters,
        "centers": centers
    }