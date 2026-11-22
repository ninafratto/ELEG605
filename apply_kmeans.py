#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nefratto

Workflow module: K-means clustering head on MoCo features, evaluated with K-fold CV on the dev set.
All values (paths, columns, PCA size, K, n_init, seed) are supplied by a conductor script.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def load_split_features(*, feature_prefix, csv_path, name_col, split_tag):
    """
    Loads <feature_prefix>_{X,y,names}.npy and returns the rows belonging to `split_tag`,
    aligned row-for-row with the returned master-split dataframe (which carries the 'fold' column).
    """
    X_all = np.load(f"{feature_prefix}_X.npy")
    y_all = np.load(f"{feature_prefix}_y.npy")
    names_all = np.load(f"{feature_prefix}_names.npy")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()

    name_to_idx = {str(name): i for i, name in enumerate(names_all)}
    split_df = df[(df['split'] == split_tag) & df[name_col].isin(set(name_to_idx))].reset_index(drop=True)
    indices = [name_to_idx[n] for n in split_df[name_col]]

    return X_all[indices], y_all[indices], split_df


def cached_features_match_split(*, feature_prefix, csv_path, name_col, split_tag):
    """True only if <feature_prefix>_{X,y,names}.npy exist AND contain exactly the samples of `split_tag` in the master CSV."""
    files = [Path(f"{feature_prefix}_{k}.npy") for k in ("X", "y", "names")]
    if not all(f.exists() for f in files):
        return False

    cached_names = {str(n) for n in np.load(files[2])}
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    expected = set(df.loc[df['split'] == split_tag, name_col].astype(str).str.strip())

    if cached_names != expected:
        print(f"[NOTICE] cached '{feature_prefix}_*.npy' does not match the current '{split_tag}' split "
              f"({len(cached_names)} cached vs {len(expected)} expected); it will be re-extracted.")
        return False
    return True


def evaluate_kmeans_on_test(*, dev_prefix, test_prefix, csv_path, name_col, n_components, n_clusters, n_init, random_state):
    """Test: fits scaler + PCA + K-means on the full dev set, assigns held-out test samples to the learned centroids."""
    X_dev, y_dev, _ = load_split_features(feature_prefix=dev_prefix, csv_path=csv_path, name_col=name_col, split_tag="dev")
    X_test, y_test, test_df = load_split_features(feature_prefix=test_prefix, csv_path=csv_path, name_col=name_col, split_tag="test")
    print(f"K-means: fitting on {len(X_dev)} dev samples, evaluating on {len(X_test)} test samples")

    if n_clusters is None:
        n_clusters = len(np.unique(y_dev))

    scaler = StandardScaler().fit(X_dev)
    pca = PCA(n_components=min(n_components, X_dev.shape[1]), random_state=random_state).fit(scaler.transform(X_dev))
    Z_dev = pca.transform(scaler.transform(X_dev))
    Z_test = pca.transform(scaler.transform(X_test))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(Z_dev)
    y_pred = kmeans.predict(Z_test)

    ari = adjusted_rand_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)
    sil = silhouette_score(Z_test, y_pred) if 1 < len(np.unique(y_pred)) < len(y_pred) else float("nan")

    # majority-vote accuracy (cluster-then-label, same rule as Mapper): each cluster takes the majority label of
    # its DEV members (ties -> the tied label most common in dev); test labels are used only for scoring.
    y_dev = np.asarray(y_dev, dtype=int)
    dev_labels, dev_counts = np.unique(y_dev, return_counts=True)
    dev_order = list(dev_labels[np.argsort(-dev_counts, kind="stable")])
    dev_clusters = kmeans.labels_
    cluster_label = {}
    for c in np.unique(dev_clusters):
        labs, cnts = np.unique(y_dev[dev_clusters == c], return_counts=True)
        tied = labs[cnts == cnts.max()]
        cluster_label[c] = min(tied, key=dev_order.index)
    y_vote = np.array([cluster_label[c] for c in y_pred])
    acc = float(np.mean(y_vote == np.asarray(y_test, dtype=int)))

    print(f"test majority-vote accuracy: {acc:.4f} | test ARI: {ari:.4f} | test NMI: {nmi:.4f} | "
          f"test silhouette: {sil:.4f}")
    print("cluster -> label (dev majority): " + ", ".join(f"{c}->{l}" for c, l in cluster_label.items()))
    print("\ncontingency (rows = true label, cols = K-means cluster):")
    print(pd.crosstab(pd.Series(y_test, name="true"), pd.Series(y_pred, name="cluster")))

    return {
        "method": "MoCo + K-means",
        "n_test": int(len(y_test)),
        "test_accuracy": acc,
        "test_ari": ari,
        "test_nmi": nmi,
        "test_silhouette": sil,
        "names": test_df[name_col].values,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_vote": y_vote,
    }


def evaluate_kmeans_head(*, feature_prefix, csv_path, name_col, n_components, n_clusters, n_init, random_state):
    """Evaluates K-Means clustering across K-fold cross-validation on the development set."""
    X_dev, y_dev, dev_df = load_split_features(feature_prefix=feature_prefix, csv_path=csv_path,
                                               name_col=name_col, split_tag="dev")
    print(f"Loaded Development Set Features: {X_dev.shape} (Test set ignored)")

    if n_clusters is None:
        n_clusters = len(np.unique(y_dev))

    folds = sorted(dev_df['fold'].unique())
    results = []

    for i, fold in enumerate(folds):
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

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        kmeans.fit(X_train_pca)

        preds_val = kmeans.predict(X_val_pca)

        ari = adjusted_rand_score(y_val, preds_val)
        nmi = normalized_mutual_info_score(y_val, preds_val)
        sil = silhouette_score(X_val_pca, preds_val)

        results.append({
            "Fold": i + 1,
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Silhouette": round(sil, 4)
        })

    results_df = pd.DataFrame(results)
    print(f"\n=== K-Means {len(folds)}-Fold Cross-Validation Benchmark (Dev Set) ===")
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
