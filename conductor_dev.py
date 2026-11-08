#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:38:52 2026

@author: nefratto
"""

from itertools import product
from pathlib import Path

from master_train_test_split import create_master_splits
from extract_features import extract_features_for_split
from multinomial_logistic_regression import (
    load_master_split_labels,
    extract_dataset_features_by_split,
    train_and_evaluate_multinomial_logistic_regression
)
from apply_kmeans import evaluate_kmeans_head
from apply_mapper import run_mapper_pipeline, run_5fold_cv_mapper_evaluation


CONFIG = {
    "csv_path": "./labels.csv",
    "master_csv": "./master_dataset_split.csv",
    "images_dir": "./Images_set",
    "name_col": "labels",
    "target_col": "continuous phases",
    "random_seed": 42,

    # --- PIPELINE 1: Physics-Informed FFT + Ridge Logistic Regression ---
    "fft": {
        "use_1d_radial": True,
        "l2_lambda_grid": [0.001, 0.1, 10.0]
    },

    # --- PIPELINE 2A: MoCo + K-Means Baseline ---
    "kmeans": {
        "pca_components": 50,
        "n_clusters": 3
    },

    # --- PIPELINE 2B: MoCo + Mapper Topology ---
    "mapper": {
        "metric_pca_components": 50,
        "dbscan_metric": "cosine",
        "dbscan_min_samples": 3,
        
        "grid_search": {
            "num_cubes": [8, 12, 16],
            "perc_overlap": [0.25, 0.35, 0.45],
            "dbscan_eps": [0.8, 1.2, 1.5]
        }
    }
}


def main():
    seed = CONFIG["random_seed"]

    print("=== [1/5] Preparing Master Dataset Splits ===")
    create_master_splits(
        csv_path=CONFIG["csv_path"],
        output_path=CONFIG["master_csv"],
        name_col=CONFIG["name_col"],
        target_col=CONFIG["target_col"],
        seed=seed
    )

    print("\n=== [2/5] Extracting MoCo Features (Dev Set Only) ===")
    if not Path("moco_features_dev_X.npy").exists():
        extract_features_for_split(split_tag="dev")
    else:
        print("[NOTICE] 'moco_features_dev_X.npy' already exists. Skipping extraction.")

    print("\n=== [3/5] Tuning FFT Ridge Logistic Regression on Dev Set (5-Fold CV) ===")
    labels_dict, df_metadata = load_master_split_labels(
        csv_path=CONFIG["master_csv"],
        name_col=CONFIG["name_col"],
        label_col=CONFIG["target_col"]
    )
    
    X_all_fft, feature_df_fft = extract_dataset_features_by_split(
        images_dir=CONFIG["images_dir"],
        label_map=labels_dict,
        df_metadata=df_metadata,
        name_col=CONFIG["name_col"],
        use_1d_radial=CONFIG["fft"]["use_1d_radial"]
    )

    best_loss = float('inf')
    best_lambda = None
    for l2_lambda in CONFIG["fft"]["l2_lambda_grid"]:
        print(f"\n--- Testing L2 Lambda = {l2_lambda} ---")
        cv_loss, cv_acc = train_and_evaluate_multinomial_logistic_regression(
            X_all=X_all_fft,
            feature_df=feature_df_fft,
            label_col=CONFIG["target_col"],
            l2_lambda=l2_lambda,
            random_state=seed
        )
        if cv_loss < best_loss:
            best_loss = cv_loss
            best_lambda = l2_lambda

    print(f"\n[SUMMARY] Optimal Dev Lambda: {best_lambda} (CV Log Loss: {best_loss:.4f})")

    print("\n=== [4/5] Running K-Means Baseline on Dev Set ===")
    evaluate_kmeans_head(
        n_components=CONFIG["kmeans"]["pca_components"],
        n_clusters=CONFIG["kmeans"]["n_clusters"],
        prefix="moco_features_dev",
        random_state=seed
    )

    print("\n=== [5/5] Running Mapper Topological CV on Dev Set ===")
    run_5fold_cv_mapper_evaluation(
        csv_path=CONFIG["master_csv"],
        feature_prefix="moco_features_dev"
    )


if __name__ == "__main__":
    main()