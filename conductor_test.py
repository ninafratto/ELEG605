#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nefratto

Conductor: final held-out TEST evaluation.
EVERY value lives in CONFIG below; the imported modules only perform the workflow.

Workflow:
  1. run conductor_dev.py (creates the dev/test split and the results_dev/ tuning tables + plots)
  2. review results_dev/ and choose ONE value per hyperparameter
  3. enter those values below and run this script ONCE. Do not re-tune based on test results.

Each model is refit on the FULL dev set (all folds) with the chosen values, then scored once on the test split.
"""

from pathlib import Path

import pandas as pd

from multinomial_logistic_regression import (
    load_master_split_labels,
    extract_dataset_features_by_split,
    fit_multinomial_logistic_regression,
    evaluate_multinomial_logistic_regression_on_test
)
from extract_features import extract_features_for_split
from apply_kmeans import evaluate_kmeans_on_test, cached_features_match_split
from apply_mapper import evaluate_mapper_on_test, run_mapper_subsample_stability
from runtime_log import RuntimeLog


CONFIG = {
    # --- data & files (must match conductor_dev.py) ---
    "paths": {
        "master_csv": "./master_dataset_split.csv",          # created by conductor_dev.py; not regenerated here
        "images_dir": "./Images_set",
        "moco_weights": "./moco_v2_800ep_pretrain.pth.tar",
        "feature_prefix": "moco_features",                   # -> moco_features_<split>_{X,y,names}.npy
        "output_dir": "./results_test",
    },
    "columns": {"name": "labels", "target": "continuous phases"},
    "plots": {"show": False, "dpi": 300},
    "random_seed": 42,          # K-means / PCA / solver seed (varied by conductor_test_seeds.py); the split is fixed by dev

    # ======== FINAL HYPERPARAMETERS: chosen from results_dev/, one value each ========

    # --- PIPELINE 1: FFT + L2 multinomial logistic regression ---  (results_dev/fft_lambda_grid.csv)
    "fft": {
        "use_1d_radial": True,
        "l2_lambda": 10.0,          # PLACEHOLDER: lowest CV cross-entropy in the 11/11 dev run; confirm
        "solver": "lbfgs",
        "max_iter": 1000,
    },

    # --- MoCo-v2 ResNet-50 feature backbone ---
    "moco": {
        "in_channels": 3,
        "batch_size": 32,
        "num_workers": 0,
        "device": "auto",
        "reuse_cached_features": True,   # stale caches (wrong split) are detected and re-extracted automatically
    },

    # --- PIPELINE 2A: K-means ---  (results_dev/kmeans_cv.csv)
    "kmeans": {"pca_components": 2, "n_clusters": 3, "n_init": 10},

    # --- PIPELINE 2B: Mapper ---  (results_dev/mapper_grid_search.csv)
    "mapper": {
        "params": {
            "metric_pca_components": 50,
            "lens_pca_components": 2,
            "num_cubes": 12,            # PLACEHOLDER: set from mapper_grid_search.csv
            "perc_overlap": 0.5,       # PLACEHOLDER
            "dbscan_eps": 0.4,          # PLACEHOLDER
            "dbscan_min_samples": 3,
            "dbscan_metric": "cosine",
        },
        "write_html": True,
        # stability of the test-only graph: n_draws subsamples of the TEST set without replacement (stratified),
        # each graph compared with the full test graph (co-membership Jaccard). Not folds: every pipeline still
        # runs on the whole test set; the subsamples only measure stability + connectedness with error.
        "resample": {"n_draws": 100, "fraction": 0.8, "stratify": True},
        "subsample_plot_style": {"marker": "o", "color": "#00539f"},
    },
}


def main():
    run_test(CONFIG)


def run_test(CONFIG):
    """Full test workflow for one config. Returns the one-row-per-method summary so other conductors can aggregate."""
    seed = CONFIG["random_seed"]
    paths, cols = CONFIG["paths"], CONFIG["columns"]
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(paths["master_csv"]).exists():
        raise FileNotFoundError(f"'{paths['master_csv']}' not found. Run conductor_dev.py first to create the dev/test split.")

    timer = RuntimeLog()

    print("=== [1/4] MoCo features (dev + test) ===")
    prefixes = {}
    for split_tag in ("dev", "test"):
        prefix = f"{paths['feature_prefix']}_{split_tag}"
        prefixes[split_tag] = prefix
        if CONFIG["moco"]["reuse_cached_features"] and cached_features_match_split(
                feature_prefix=prefix, csv_path=paths["master_csv"], name_col=cols["name"], split_tag=split_tag):
            print(f"skipping extraction: '{prefix}_*.npy' already matches the current {split_tag} split")
        else:
            with timer.stage(f"MoCo feature extraction ({split_tag})"):
                extract_features_for_split(split_tag=split_tag, images_dir=paths["images_dir"], master_csv=paths["master_csv"],
                                           weights_path=paths["moco_weights"], output_prefix=prefix,
                                           name_col=cols["name"], label_col=cols["target"],
                                           in_channels=CONFIG["moco"]["in_channels"], batch_size=CONFIG["moco"]["batch_size"],
                                           num_workers=CONFIG["moco"]["num_workers"], device=CONFIG["moco"]["device"])

    print("\n=== [2/4] FFT logistic regression: refit on dev, score on test ===")
    with timer.stage("FFT feature extraction"):
        labels_dict, df_metadata = load_master_split_labels(csv_path=paths["master_csv"], name_col=cols["name"],
                                                            label_col=cols["target"])
        X_all_fft, feature_df_fft = extract_dataset_features_by_split(images_dir=paths["images_dir"], label_map=labels_dict,
                                                                      df_metadata=df_metadata, name_col=cols["name"],
                                                                      use_1d_radial=CONFIG["fft"]["use_1d_radial"])
    with timer.stage("FFT logreg fit + test"):
        fft_model = fit_multinomial_logistic_regression(X_all=X_all_fft, feature_df=feature_df_fft, label_col=cols["target"],
                                                        l2_lambda=CONFIG["fft"]["l2_lambda"], solver=CONFIG["fft"]["solver"],
                                                        max_iter=CONFIG["fft"]["max_iter"], random_state=seed)
        fft_res = evaluate_multinomial_logistic_regression_on_test(model=fft_model, X_all=X_all_fft, feature_df=feature_df_fft,
                                                                   label_col=cols["target"], name_col=cols["name"])

    print("\n=== [3/4] K-means: fit on dev, assign test ===")
    with timer.stage("K-means fit + test"):
        km_res = evaluate_kmeans_on_test(dev_prefix=prefixes["dev"], test_prefix=prefixes["test"], csv_path=paths["master_csv"],
                                         name_col=cols["name"], n_components=CONFIG["kmeans"]["pca_components"],
                                         n_clusters=CONFIG["kmeans"]["n_clusters"], n_init=CONFIG["kmeans"]["n_init"],
                                         random_state=seed)

    print("\n=== [4/4] Mapper: test topology + majority-vote labels ===")
    with timer.stage("Mapper fit + test"):
        mp_res = evaluate_mapper_on_test(dev_prefix=prefixes["dev"], test_prefix=prefixes["test"], csv_path=paths["master_csv"],
                                         name_col=cols["name"], mapper_params=CONFIG["mapper"]["params"],
                                         random_state=seed,
                                         output_html=out_dir / "mapper_test_graph.html" if CONFIG["mapper"]["write_html"] else None)
    with timer.stage("Mapper test subsample stability"):
        _, mp_sub = run_mapper_subsample_stability(
            feature_prefix=prefixes["test"], csv_path=paths["master_csv"], name_col=cols["name"], split_tag="test",
            mapper_params=CONFIG["mapper"]["params"], random_state=seed, resample=CONFIG["mapper"]["resample"],
            plot_style=CONFIG["mapper"]["subsample_plot_style"], plot_path=out_dir / "mapper_test_subsamples.png",
            show_plot=CONFIG["plots"]["show"], dpi=CONFIG["plots"]["dpi"])
    mp_res.update({f"sub_{k}": v for k, v in mp_sub.items() if not k.startswith("ref_")})   # ref_* == topo_*

    # summary table (one row per method) + per-sample predictions
    per_sample = ("names", "y_true", "y_pred", "y_vote")
    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in per_sample} for r in (fft_res, km_res, mp_res)])
    summary.to_csv(out_dir / "test_summary.csv", index=False)

    preds = pd.DataFrame({cols["name"]: fft_res["names"], "true": fft_res["y_true"], "fft_logreg": fft_res["y_pred"]})
    for col, r, key in (("kmeans_cluster", km_res, "y_pred"), ("kmeans_vote", km_res, "y_vote"),
                        ("mapper_vote", mp_res, "y_pred")):
        preds = preds.merge(pd.DataFrame({cols["name"]: r["names"], col: r[key]}), on=cols["name"], how="outer")
    preds.to_csv(out_dir / "test_predictions.csv", index=False)

    print("\n=== TEST SUMMARY ===")
    print(summary.T.to_string(header=False))
    print(f"\nsaved '{out_dir / 'test_summary.csv'}' and '{out_dir / 'test_predictions.csv'}'")

    timer.save(out_dir / "runtime.csv")
    return summary


if __name__ == "__main__":
    main()
