import numpy as np
import json
import os
import base64
import io
import random

random.seed(42)
np.random.seed(42)

def generate_dataset():
    """Generate 1000 customer records: Age & Annual Spending (USD)"""
    records = []
    # Segment 1: Young budget shoppers (age 18-28, spend 500-2000)
    for _ in range(340):
        age = round(random.uniform(18, 28), 1)
        spend = round(random.uniform(500, 2000), 2)
        records.append({"age": age, "annual_spending": spend})
    # Segment 2: Mid-career moderate spenders (age 30-45, spend 3000-6000)
    for _ in range(360):
        age = round(random.uniform(30, 45), 1)
        spend = round(random.uniform(3000, 6000), 2)
        records.append({"age": age, "annual_spending": spend})
    # Segment 3: Established high spenders (age 47-65, spend 7000-14000)
    for _ in range(300):
        age = round(random.uniform(47, 65), 1)
        spend = round(random.uniform(7000, 14000), 2)
        records.append({"age": age, "annual_spending": spend})
    return records

def run_clustering():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = generate_dataset()
    X = np.array([[r["age"], r["annual_spending"]] for r in data])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Build results
    results = []
    for i, record in enumerate(data):
        row = record.copy()
        row["cluster"] = int(labels[i])
        results.append(row)

    # Cluster summary
    summary = {}
    for k in range(3):
        cluster_data = [data[i] for i in range(len(data)) if labels[i]==k]
        summary[k] = {
            "count": len(cluster_data),
            "avg_age": round(np.mean([r["age"] for r in cluster_data]), 2),
            "avg_spending": round(np.mean([r["annual_spending"] for r in cluster_data]), 2),
        }

    centroids_info = [
        {"age": round(c[0], 2), "annual_spending": round(c[1], 2)}
        for c in centroids_original
    ]

    # Generate scatter plot as base64
    plot_b64 = generate_plot(X, labels, centroids_original)

    return {
        "results": results[:50],  # show first 50 in table
        "total": len(results),
        "summary": summary,
        "centroids": centroids_info,
        "plot": plot_b64
    }

def generate_plot(X, labels, centroids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#38bdf8', '#fb923c', '#a78bfa']
    cluster_names = ['Budget Shoppers', 'Moderate Spenders', 'High Spenders']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    # Sort clusters by centroid spending for consistent coloring
    centroid_order = np.argsort(centroids[:, 1])

    for rank, k in enumerate(centroid_order):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[rank], label=cluster_names[rank],
                   alpha=0.6, s=30, edgecolors='none')

    # Plot centroids
    for rank, k in enumerate(centroid_order):
        ax.scatter(centroids[k, 0], centroids[k, 1],
                   c='white', marker='X', s=200, zorder=5,
                   edgecolors=colors[rank], linewidths=1.5)

    ax.set_xlabel('Age', color='#94a3b8', fontsize=12)
    ax.set_ylabel('Annual Spending (USD)', color='#94a3b8', fontsize=12)
    ax.set_title('Customer Segmentation – K-Means (k=3)', color='#f1f5f9', fontsize=14, fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#334155')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9', fontsize=10)
    ax.grid(True, color='#334155', alpha=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
