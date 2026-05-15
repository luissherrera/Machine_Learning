import math
import json

# ─────────────────────────────────────────────
# Dataset: 100 records – Age & Monthly Income
# ─────────────────────────────────────────────
RAW_DATA = [
    # Young / low income (cluster A)
    [18,800],[19,850],[20,900],[21,950],[22,1000],[21,980],[20,870],[19,830],[18,810],[22,1020],
    [21,960],[20,910],[19,860],[18,820],[22,1050],[21,990],[20,920],[19,840],[18,790],[22,1010],
    [21,970],[20,930],[19,870],[18,800],[22,1030],[21,950],[20,890],[19,840],[18,815],[22,1040],
    [21,985],[20,905],[19,855],[18,805],
    # Middle-aged / mid income (cluster B)
    [35,2800],[36,2900],[37,3000],[38,3100],[39,3200],[36,2850],[37,2950],[38,3050],[39,3150],[35,2750],
    [36,2820],[37,2920],[38,3020],[39,3120],[35,2780],[36,2880],[37,2980],[38,3080],[39,3180],[35,2760],
    [36,2860],[37,2960],[38,3060],[39,3160],[35,2790],[36,2890],[37,2990],[38,3090],[39,3190],[35,2770],
    [36,2870],[37,2970],
    # Senior / high income (cluster C)
    [52,5500],[53,5600],[54,5700],[55,5800],[56,5900],[52,5450],[53,5550],[54,5650],[55,5750],[56,5850],
    [52,5480],[53,5580],[54,5680],[55,5780],[56,5880],[52,5460],[53,5560],[54,5660],[55,5760],[56,5860],
    [52,5490],[53,5590],[54,5690],[55,5790],[56,5890],[52,5470],[53,5570],[54,5670],[55,5770],[56,5870],
    [52,5485],[53,5585],[54,5685],
]

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def compute_centroid(points):
    n = len(points)
    return [sum(p[0] for p in points)/n, sum(p[1] for p in points)/n]

def compute_variance(data, labels, centroids):
    total = 0
    for i, point in enumerate(data):
        c = centroids[labels[i]]
        total += euclidean(point, c)**2
    return round(total, 2)

def run_iterations():
    data = RAW_DATA
    # Initial centroids: manually chosen representative points
    centroids = [[20, 900], [37, 3000], [54, 5650]]
    
    iterations = []
    for it in range(3):
        # Assign clusters
        labels = []
        rows = []
        for point in data:
            dists = [round(euclidean(point, c), 2) for c in centroids]
            assigned = dists.index(min(dists))
            labels.append(assigned)
            rows.append({
                "age": point[0], "income": point[1],
                "d0": dists[0], "d1": dists[1], "d2": dists[2],
                "cluster": assigned
            })
        
        variance = compute_variance(data, labels, centroids)
        
        # Update centroids
        new_centroids = []
        for k in range(3):
            cluster_points = [data[i] for i in range(len(data)) if labels[i]==k]
            if cluster_points:
                new_centroids.append([round(v,2) for v in compute_centroid(cluster_points)])
            else:
                new_centroids.append(centroids[k])
        
        iterations.append({
            "iteration": it+1,
            "centroids_before": [[round(c[0],2), round(c[1],2)] for c in centroids],
            "centroids_after": new_centroids,
            "variance": variance,
            "cluster_counts": [labels.count(0), labels.count(1), labels.count(2)],
            "rows": rows
        })
        centroids = new_centroids
    
    return iterations

ITERATIONS = run_iterations()
VARIANCES = [it["variance"] for it in ITERATIONS]

def get_manual_data():
    return {
        "dataset": RAW_DATA,
        "iterations": ITERATIONS,
        "variances": VARIANCES
    }
