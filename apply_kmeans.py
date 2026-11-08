#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:03:25 2026

@author: nefratto
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def load_dev_split_data(prefix="moco_features_dev", csv_path="master_dataset_split.csv"):
    """Loads feature matrix and filters strictly for the development set."""
    X_all = np.load(f"{prefix}_X.npy")
    y_all = np.load(f"{prefix}_y.npy")
    names_all = np.load(f"{prefix}_names.npy")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    name_to_idx = {name: i for i, name in enumerate(names_all)}
    dev_df = df[df['split'] == 'dev'].reset_index(drop=True)
    dev_indices = [name_to_idx[n] for n in dev_df['labels'] if n in name_to_idx]

    return X_all[dev_indices], y_all[dev_indices], dev_df


def evaluate_kmeans_head(n_components=50, n_clusters=3, prefix="moco_features_dev", random_state=42):
    """Evaluates K-Means clustering across 5-fold cross-validation on the development set."""
    X_dev, y_dev, dev_df = load_dev_split_data(prefix=prefix)
    print(f"Loaded Development Set Features: {X_dev.shape} (Test set ignored)")

    if n_clusters is None:
        n_clusters = len(np.unique(y_dev))

    results = []
    
    for fold in range(5):
        train_fold_mask = (dev_df['fold'] != fold).values
        val_fold_mask = (dev_df['fold'] == fold).values

        X_train, y_train = X_dev[train_fold_mask], y_dev[train_fold_mask]
        X_val, y_val = X_dev[val_fold_mask], y_dev[val_fold_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        pca = PCA(n_components=min(n_components, X_train.shape[1]), random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        kmeans.fit(X_train_pca)

        preds_val = kmeans.predict(X_val_pca)

        ari = adjusted_rand_score(y_val, preds_val)
        nmi = normalized_mutual_info_score(y_val, preds_val)
        sil = silhouette_score(X_val_pca, preds_val)

        results.append({
            "Fold": fold + 1,
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette": round(sil, 4)
        })

    results_df = pd.DataFrame(results)
    print("\n=== K-Means 5-Fold Cross-Validation Benchmark (Dev Set) ===")
    print(results_df.to_string(index=False))

    # Calculate Means and Standard Errors of the Mean (SEM)
    ari_mean, ari_sem = results_df['ARI'].mean(), results_df['ARI'].sem()
    nmi_mean, nmi_sem = results_df['NMI'].mean(), results_df['NMI'].sem()
    sil_mean, sil_sem = results_df['Silhouette'].mean(), results_df['Silhouette'].sem()

    print("\n--- Benchmark Summary (Mean ± SEM) ---")
    print(f"ARI:        {ari_mean:.4f} ± {ari_sem:.4f}")
    print(f"NMI:        {nmi_mean:.4f} ± {nmi_sem:.4f}")
    print(f"Silhouette: {sil_mean:.4f} ± {sil_sem:.4f}")

    return results_df


if __name__ == "__main__":
    start_time = time.perf_counter()
    evaluate_kmeans_head(n_components=50, prefix="moco_features_dev", random_state=42)
    elapsed_time = time.perf_counter() - start_time
    print(f"\n[INFO] Execution time: {elapsed_time:.2f} seconds.")
