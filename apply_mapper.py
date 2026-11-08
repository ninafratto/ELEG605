#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:06:40 2026

@author: nefratto
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def load_dev_moco_features(prefix="moco_features_dev", csv_path="master_dataset_split.csv"):
    """Loads extracted feature matrices strictly for the development set ('dev')."""
    X_all = np.load(f"{prefix}_X.npy")
    y_all = np.load(f"{prefix}_y.npy")
    names_all = np.load(f"{prefix}_names.npy")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    name_to_idx = {name: i for i, name in enumerate(names_all)}
    dev_df = df[df['split'] == 'dev']
    dev_indices = [name_to_idx[n] for n in dev_df['labels'] if n in name_to_idx]

    return X_all[dev_indices], y_all[dev_indices]


def kmapper_to_networkx(graph):
    """Converts a KeplerMapper output dictionary into a NetworkX Graph object."""
    G = nx.Graph()
    for node_id in graph['nodes']:
        G.add_node(node_id)
    for u, neighbors in graph['links'].items():
        for v in neighbors:
            G.add_edge(u, v)
    return G


def compute_mapper_topological_metrics(graph, X_metric):
    """Computes Betti numbers (b0, b1), graph connectedness, singletons, and stability metrics."""
    G = kmapper_to_networkx(graph)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {
            "b0": 0, "b1": 0, "connectedness": 0.0, 
            "singletons": 0, "stability": 0.0, "mean_silhouette": 0.0
        }

    b0 = nx.number_connected_components(G)
    b1 = num_edges - num_nodes + b0

    largest_cc_size = len(max(nx.connected_components(G), key=len))
    connectedness = largest_cc_size / num_nodes if num_nodes > 0 else 0.0
    singletons = sum(1 for node, degree in G.degree() if degree == 0)
    stability = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

    cluster_silhouette_scores = []
    for node_id, point_indices in graph['nodes'].items():
        if len(point_indices) > 1:
            sub_X = X_metric[point_indices]
            db = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(sub_X)
            if len(set(db.labels_)) > 1 and len(set(db.labels_)) != len(sub_X):
                score = silhouette_score(sub_X, db.labels_)
                cluster_silhouette_scores.append(score)

    mean_silhouette = float(np.mean(cluster_silhouette_scores)) if cluster_silhouette_scores else 0.0

    return {
        "b0": b0,
        "b1": b1,
        "connectedness": connectedness,
        "singletons": singletons,
        "stability": stability,
        "mean_silhouette": mean_silhouette
    }


def plot_cv_fold_mapper_metrics(fold_metrics_df):
    """Plots topological evaluation metrics across each of the 5 cross-validation folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    folds = fold_metrics_df['Fold'].values

    # Plot 1: Betti Numbers (b0, b1) and Topological Stability
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_xlabel('CV fold number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Betti numbers', color=color, fontsize=11, fontweight='bold')
    ax1.scatter(folds, fold_metrics_df['b0'], marker='o', color='blue', label='b0 (components)')
    ax1.scatter(folds, fold_metrics_df['b1'], marker='s', color='navy', label='b1 (holes)')
    ax1.set_xticks(folds)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.5)

    ax1_twin = ax1.twinx()
    color = 'tab:red'
    ax1_twin.set_ylabel('topological stability', color=color, fontsize=11, fontweight='bold')
    ax1_twin.scatter(folds, fold_metrics_df['stability'], marker='^', color=color, label='Stability')
    ax1_twin.tick_params(axis='y', labelcolor=color)

    # Plot 2: Graph Connectedness, Singletons, and Silhouette Score
    ax2 = axes[1]
    ax2.plot(folds, fold_metrics_df['connectedness'], marker='o', color='green', label='Connectedness Ratio')
    ax2.plot(folds, fold_metrics_df['mean_silhouette'], marker='d', color='purple', linewidth=2, label='Mean Silhouette Score')
    ax2.set_xlabel('CV Fold Number', fontsize=11)
    ax2.set_ylabel('Metric Value [0, 1]', fontsize=11)
    ax2.set_xticks(folds)
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2_twin = ax2.twinx()
    color = 'tab:orange'
    ax2_twin.set_ylabel('Singleton Nodes Count', color=color, fontsize=11)
    ax2_twin.plot(folds, fold_metrics_df['singletons'], marker='s', color=color, linestyle='--', label='Singletons')
    ax2_twin.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.set_title('5-Fold CV: Graph Connectedness & Quality', fontsize=12)

    plt.tight_layout()
    plt.savefig("mapper_5fold_cv_evaluation.png", dpi=300)
    print("\nSaved 5-Fold Cross-Validation evaluation plot to 'mapper_5fold_cv_evaluation.png'")
    plt.show()


def run_mapper_pipeline(
    metric_pca_components=50, 
    num_cubes=12, 
    perc_overlap=0.35, 
    dbscan_eps=1.2, 
    dbscan_min_samples=3,
    dbscan_metric="cosine",
    output_html="moco_mapper_dev_output.html",
    prefix="moco_features_dev",
    csv_path="master_dataset_split.csv",
    random_state=42
):
    """Executes KeplerMapper strictly on the Development set."""
    X_target, y_target = load_dev_moco_features(prefix=prefix, csv_path=csv_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_target)
    
    pca_metric = PCA(n_components=min(metric_pca_components, X_target.shape[1]), random_state=random_state)
    X_metric = pca_metric.fit_transform(X_scaled)

    pca_lens = PCA(n_components=2, random_state=random_state)
    lens = MinMaxScaler().fit_transform(pca_lens.fit_transform(X_metric))

    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(n_cubes=num_cubes, perc_overlap=perc_overlap)
    clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)

    graph = mapper.map(lens=lens, X=X_metric, cover=cover, clusterer=clusterer)

    if output_html:
        mapper.visualize(
            graph,
            color_values=y_target,
            color_function_name="Continuous Phase Target",
            title="MoCo ResNet-50 Mapper Graph (Development Set)",
            path_html=output_html
        )
    
    return mapper, graph, X_metric


def plot_hyperparameter_evaluation(prefix="moco_features_dev", csv_path="master_dataset_split.csv"):
    """
    Sweeps hyperparameter ranges on the Dev set for cover resolution, overlap percentage, 
    and DBSCAN eps metric distance, plotting requested topological metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Sweep Cover Intervals (num_cubes): Topological Stability & Betti Numbers
    print("\nSweeping Cover Interval Resolution (num_cubes)...")
    cubes_range = np.arange(4, 20, 2)
    b0_list, b1_list, stability_list = [], [], []

    for n_c in cubes_range:
        _, graph, X_metric = run_mapper_pipeline(num_cubes=n_c, output_html=None, prefix=prefix, csv_path=csv_path)
        metrics = compute_mapper_topological_metrics(graph, X_metric)
        b0_list.append(metrics['b0'])
        b1_list.append(metrics['b1'])
        stability_list.append(metrics['stability'])

    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_xlabel('Cover Resolution (num_cubes)', fontsize=11)
    ax1.set_ylabel('Betti Numbers (b0: components, b1: loops)', color=color, fontsize=11)
    ax1.plot(cubes_range, b0_list, marker='o', color='blue', label='b0 (Components)')
    ax1.plot(cubes_range, b1_list, marker='s', color='navy', linestyle='--', label='b1 (1D Cycles)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1_twin = ax1.twinx()
    color = 'tab:red'
    ax1_twin.set_ylabel('Topological Stability Metric', color=color, fontsize=11)
    ax1_twin.plot(cubes_range, stability_list, marker='^', color=color, label='Stability')
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Cover Intervals vs. Topology & Stability', fontsize=12)

    # 2. Sweep Overlap Percentage (perc_overlap): Connectedness & Singletons
    print("Sweeping Overlap Percentage (perc_overlap)...")
    overlap_range = np.linspace(0.10, 0.60, 9)
    conn_list, singletons_list = [], []

    for ov in overlap_range:
        _, graph, X_metric = run_mapper_pipeline(perc_overlap=ov, output_html=None, prefix=prefix, csv_path=csv_path)
        metrics = compute_mapper_topological_metrics(graph, X_metric)
        conn_list.append(metrics['connectedness'])
        singletons_list.append(metrics['singletons'])

    ax2 = axes[1]
    color = 'tab:green'
    ax2.set_xlabel('Overlap Percentage (perc_overlap)', fontsize=11)
    ax2.set_ylabel('Graph Connectedness Ratio', color=color, fontsize=11)
    ax2.plot(overlap_range, conn_list, marker='o', color=color, label='Connectedness')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2_twin = ax2.twinx()
    color = 'tab:orange'
    ax2_twin.set_ylabel('Singleton Nodes Count', color=color, fontsize=11)
    ax2_twin.plot(overlap_range, singletons_list, marker='s', color=color, linestyle='--', label='Singletons')
    ax2_twin.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Overlap % vs. Connectedness & Singletons', fontsize=12)

    # 3. Sweep Clusterer Distance (dbscan_eps): Mean Silhouette Score
    print("Sweeping Clusterer Distance Threshold (dbscan_eps)...")
    eps_range = np.linspace(0.2, 1.8, 9)
    silhouette_list = []

    for eps in eps_range:
        _, graph, X_metric = run_mapper_pipeline(dbscan_eps=eps, output_html=None, prefix=prefix, csv_path=csv_path)
        metrics = compute_mapper_topological_metrics(graph, X_metric)
        silhouette_list.append(metrics['mean_silhouette'])

    ax3 = axes[2]
    ax3.plot(eps_range, silhouette_list, marker='d', color='purple', linewidth=2, label='Mean Silhouette')
    ax3.set_xlabel('DBSCAN Distance Threshold (eps)', fontsize=11)
    ax3.set_ylabel('Mean Cluster Silhouette Score', fontsize=11)
    ax3.set_title('Clusterer Distance vs. Silhouette Score', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("mapper_hyperparameter_evaluation.png", dpi=300)
    print("\nSaved hyperparameter evaluation plot to 'mapper_hyperparameter_evaluation.png'")
    plt.show()


def run_5fold_cv_mapper_evaluation(csv_path="master_dataset_split.csv", feature_prefix="moco_features_dev"):
    """Executes 5-Fold Cross Validation over the development set and evaluates metrics per fold."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    X_all = np.load(f"{feature_prefix}_X.npy")
    y_all = np.load(f"{feature_prefix}_y.npy")
    names_all = np.load(f"{feature_prefix}_names.npy")

    name_to_idx = {name: i for i, name in enumerate(names_all)}
    dev_df = df[df['split'] == 'dev'].reset_index(drop=True)
    dev_indices = [name_to_idx[n] for n in dev_df['labels'] if n in name_to_idx]

    X_dev = X_all[dev_indices]
    y_dev = y_all[dev_indices]

    print(f"\nExecuting 5-Fold Cross-Validation Mapper Evaluation (Dev set size: {len(dev_df)})...")
    cv_results = []
    
    for fold in range(5):
        val_fold_mask = (dev_df['fold'] == fold).values
        X_val_fold = X_dev[val_fold_mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_val_fold)
        pca_metric = PCA(n_components=min(50, X_val_fold.shape[1]), random_state=42)
        X_metric = pca_metric.fit_transform(X_scaled)

        pca_lens = PCA(n_components=2, random_state=42)
        lens = MinMaxScaler().fit_transform(pca_lens.fit_transform(X_metric))

        mapper = km.KeplerMapper(verbose=0)
        cover = km.Cover(n_cubes=12, perc_overlap=0.35)
        clusterer = DBSCAN(eps=1.2, min_samples=1, metric="cosine")

        graph = mapper.map(lens=lens, X=X_metric, cover=cover, clusterer=clusterer)
        metrics = compute_mapper_topological_metrics(graph, X_metric)
        metrics['Fold'] = fold + 1
        cv_results.append(metrics)
        print(f"Fold {fold + 1}/5 | b0: {metrics['b0']:2d} | b1: {metrics['b1']:2d} | "
              f"Connectedness: {metrics['connectedness']:.3f} | Singletons: {metrics['singletons']:2d} | "
              f"Silhouette: {metrics['mean_silhouette']:.3f}")

    metrics_df = pd.DataFrame(cv_results)
    
    # Generate 5-fold CV plot
    plot_cv_fold_mapper_metrics(metrics_df)
    return metrics_df


if __name__ == "__main__":
    # 1. Generate 5-Fold Cross Validation fold plot
    run_5fold_cv_mapper_evaluation()
    
    # 2. Generate Hyperparameter sweep plot
    plot_hyperparameter_evaluation()