#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nefratto

Conductor: development-set experiments (hyperparameter tuning via K-fold CV).
EVERY tunable value lives in CONFIG below; the imported modules only perform the workflow.
"""

from itertools import product
from pathlib import Path

import pandas as pd

from master_train_test_split import create_master_splits
from multinomial_logistic_regression import (load_master_split_labels, extract_dataset_features_by_split,
                                             train_and_evaluate_multinomial_logistic_regression,
                                             plot_cv_lambda_comparison)
from extract_features import extract_features_for_split
from apply_kmeans import evaluate_kmeans_head, cached_features_match_split
from apply_mapper import (run_mapper_subsample_stability, run_mapper_pipeline, plot_mapper_grid_comparison,
                          plot_mapper_grid_interaction, plot_mapper_grid_heatmap)
from runtime_log import RuntimeLog


CONFIG = {
    # --- data & files ---
    "paths": {
        "labels_csv": "./labels.csv",                         # raw labels (input to the split)
        "master_csv": "./master_dataset_split.csv",           # dev/test + fold assignments (output of the split)
        "images_dir": "./Images_set",
        "moco_weights": "./moco_v2_800ep_pretrain.pth.tar",
        "feature_prefix": "moco_features",                    # -> moco_features_<split>_{X,y,names}.npy
        "output_dir": "./results_dev",                        # plots + summary CSVs
    },
    "columns": {"name": "labels", "target": "continuous phases"},
    "random_seed": 42,          # folds, logistic regression, K-means, Mapper subsamples (varied by conductor_dev_seeds.py)
    "plots": {"show": False, "dpi": 300},                     # show=True pops up every figure (blocks grid loops)

    # --- dev/test split ---
    "split": {"test_size": 0.20, "n_folds": 5, "seed": 42},   # seed fixes WHICH images are test; keep it constant

    # --- PIPELINE 1: physics-informed FFT + L2 multinomial logistic regression ---
    "fft": {
        "use_1d_radial": True,
        "l2_lambda_grid": [0.001, 0.1, 10.0],
        "solver": "lbfgs",
        "max_iter": 1000,
        # per-fold comparison figures (accuracy subplot + cross-entropy subplot): each entry is a list of lambdas
        # (from l2_lambda_grid) drawn on one figure; compare_styles[i] is used for the i-th lambda.
        # Empty list -> no comparison figures.
        "compare_lambdas": [[0.001, 0.1, 10.0]],
        "compare_styles": [
            {"marker": "o", "color": "#00539f"},
            {"marker": "^", "color": "#ffd200"},
            {"marker": "s", "color": "#bdbdbd"},
        ],
        "colors": {"loss": "#ffd200", "accuracy": "#00539f"},   # single-lambda CV plots
        "baseline_accuracy": 0.789,   # majority-class baseline drawn as a dashed line on the accuracy axis; None -> no line
        "baseline_color": "black",    # baseline line color in the comparison figure
    },

    # --- MoCo-v2 ResNet-50 feature backbone ---
    "moco": {
        "in_channels": 3,
        "batch_size": 32,
        "num_workers": 0,
        "device": "auto",                  # "auto", "cpu", or "cuda"
        "reuse_cached_features": True,     # cache is auto-invalidated if the split changes; set False after changing images or weights
    },

    # --- PIPELINE 2A: K-means baseline ---
    "kmeans": {"pca_components": 2, "n_clusters": 3, "n_init": 10},   # n_clusters=None -> number of label classes

    # --- PIPELINE 2B: Mapper ---
    "mapper": {
        # held fixed across the grid (any key here can be moved into grid_search instead)
        "fixed": {
            "metric_pca_components": 50,
            "lens_pca_components": 2,
            "dbscan_metric": "cosine",
            "dbscan_min_samples": 3,
        },
        # full factorial grid
        "grid_search": {
            "num_cubes": [12, 24],
            "perc_overlap": [0.35, 0.50],
            "dbscan_eps": [0.15, 0.4],
        },
        "write_html": True,     # full-dev-set graph per grid combination, colored by PC1 (no labels)
        # stability of the full-dev graph: n_draws subsamples of the dev set, drawn WITHOUT replacement (fraction of
        # the points; stratified by label if stratify). Stability = co-membership Jaccard of each subsample graph vs
        # the full-dev graph; every metric is reported as mean +/- std across subsamples.
        "resample": {"n_draws": 100, "fraction": 0.8, "stratify": True},
        # per-combination plot (stability + connectedness for each subsample)
        "subsample_plot_style": {"marker": "o", "color": "#00539f"},
        # all-combinations comparison plot: compare_styles[i] is used for the i-th grid combination
        "compare_grid": True,
        "compare_styles": [
            {"marker": "o", "color": "#00539f"},
            {"marker": "^", "color": "#ffd200"},
            {"marker": "s", "color": "#bdbdbd"},
            {"marker": "D", "color": "#c8102e"},
            {"marker": "v", "color": "#00843d"},
            {"marker": "P", "color": "#7f3f98"},
            {"marker": "X", "color": "#f28e2b"},
            {"marker": "*", "color": "#000000"},
        ],
        # axis/legend names for the grid hyperparameters
        "param_labels": {"num_cubes": "cubes", "perc_overlap": "overlap", "dbscan_eps": "eps"},
        # option 1: interaction plot (x axis, marker/color, filled vs open)
        "interaction_plot": {
            "enabled": True,
            "x_param": "perc_overlap",
            "color_param": "num_cubes",
            "fill_param": "dbscan_eps",     # 1st (smallest) value filled, others open
            "color_styles": [{"marker": "o", "color": "#00539f"}, {"marker": "^", "color": "#ffd200"}],
            "x_offset": 0.04,
        },
        # option 2: heatmaps (rows x columns of each panel; one panel per value of panel_param)
        "heatmap_plot": {
            "enabled": True,
            "row_param": "num_cubes",
            "col_param": "perc_overlap",
            "panel_param": "dbscan_eps",
            "cmap": "Blues",
        },
    },
}


def _fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def main():
    run_dev(CONFIG)


def run_dev(CONFIG):
    """Full dev workflow for one config. Returns (fft_df, mapper_summary) so other conductors can aggregate runs."""
    seed = CONFIG["random_seed"]
    paths, cols = CONFIG["paths"], CONFIG["columns"]
    plot_kw = {"show_plot": CONFIG["plots"]["show"], "dpi": CONFIG["plots"]["dpi"]}
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("conducting development...")
    timer = RuntimeLog()

    # dev/test split
    with timer.stage("dev/test split"):
        create_master_splits(csv_path=paths["labels_csv"], output_path=paths["master_csv"],
                             name_col=cols["name"], target_col=cols["target"],
                             test_size=CONFIG["split"]["test_size"], n_folds=CONFIG["split"]["n_folds"],
                             seed=CONFIG["split"]["seed"], fold_seed=seed)

    # FFT logistic regression: lambda grid
    with timer.stage("FFT feature extraction"):
        labels_dict, df_metadata = load_master_split_labels(csv_path=paths["master_csv"], name_col=cols["name"],
                                                            label_col=cols["target"])
        X_all_fft, feature_df_fft = extract_dataset_features_by_split(images_dir=paths["images_dir"], label_map=labels_dict,
                                                                      df_metadata=df_metadata, name_col=cols["name"],
                                                                      use_1d_radial=CONFIG["fft"]["use_1d_radial"])
    fft_colors = {"loss_color": CONFIG["fft"]["colors"]["loss"], "accuracy_color": CONFIG["fft"]["colors"]["accuracy"],
                  "baseline_accuracy": CONFIG["fft"]["baseline_accuracy"]}
    fft_rows = []
    fft_cv = {}
    for l2_lambda in CONFIG["fft"]["l2_lambda_grid"]:
        print(f"\n--- L2 lambda = {l2_lambda} ---")
        with timer.stage(f"FFT logreg CV (lambda={_fmt(l2_lambda)})"):
            fft_cv[l2_lambda] = train_and_evaluate_multinomial_logistic_regression(
                X_all=X_all_fft, feature_df=feature_df_fft, label_col=cols["target"], l2_lambda=l2_lambda,
                solver=CONFIG["fft"]["solver"], max_iter=CONFIG["fft"]["max_iter"], random_state=seed,
                plot_path=out_dir / f"fft_cv_lambda_{_fmt(l2_lambda)}.png", **fft_colors, **plot_kw)
        fft_rows.append({"l2_lambda": l2_lambda, "cv_log_loss": fft_cv[l2_lambda]["mean_log_loss"],
                         "cv_accuracy": fft_cv[l2_lambda]["mean_accuracy"], "runtime_s": timer.rows[-1]["seconds"]})

    for lambdas in CONFIG["fft"]["compare_lambdas"]:
        missing = [lam for lam in lambdas if lam not in fft_cv]
        if missing:
            raise ValueError(f"compare_lambdas {missing} are not in l2_lambda_grid {CONFIG['fft']['l2_lambda_grid']}")
        if len(lambdas) > len(CONFIG["fft"]["compare_styles"]):
            raise ValueError(f"compare_lambdas {lambdas} needs at least {len(lambdas)} compare_styles")
        plot_cv_lambda_comparison({lam: fft_cv[lam] for lam in lambdas}, styles=CONFIG["fft"]["compare_styles"],
                                  baseline_accuracy=CONFIG["fft"]["baseline_accuracy"],
                                  baseline_color=CONFIG["fft"]["baseline_color"],
                                  plot_path=out_dir / f"fft_cv_compare_{'_vs_'.join(_fmt(l) for l in lambdas)}.png",
                                  **plot_kw)

    fft_df = pd.DataFrame(fft_rows)
    fft_df.to_csv(out_dir / "fft_lambda_grid.csv", index=False)
    best = fft_df.loc[fft_df["cv_log_loss"].idxmin()]
    print(f"\n[SUMMARY] optimal dev lambda: {best['l2_lambda']} (CV log loss: {best['cv_log_loss']:.4f})")

    # CNN feature extraction
    dev_prefix = f"{paths['feature_prefix']}_dev"
    if CONFIG["moco"]["reuse_cached_features"] and cached_features_match_split(
            feature_prefix=dev_prefix, csv_path=paths["master_csv"], name_col=cols["name"], split_tag="dev"):
        print(f"skipping extraction: '{dev_prefix}_*.npy' already matches the current dev split")
    else:
        with timer.stage("MoCo feature extraction (dev)"):
            extract_features_for_split(split_tag="dev", images_dir=paths["images_dir"], master_csv=paths["master_csv"],
                                       weights_path=paths["moco_weights"], output_prefix=dev_prefix,
                                       name_col=cols["name"], label_col=cols["target"],
                                       in_channels=CONFIG["moco"]["in_channels"], batch_size=CONFIG["moco"]["batch_size"],
                                       num_workers=CONFIG["moco"]["num_workers"], device=CONFIG["moco"]["device"])

    # K-means baseline
    with timer.stage("K-means CV"):
        kmeans_df = evaluate_kmeans_head(feature_prefix=dev_prefix, csv_path=paths["master_csv"], name_col=cols["name"],
                                         n_components=CONFIG["kmeans"]["pca_components"],
                                         n_clusters=CONFIG["kmeans"]["n_clusters"],
                                         n_init=CONFIG["kmeans"]["n_init"], random_state=seed)
    kmeans_df.to_csv(out_dir / "kmeans_cv.csv", index=False)

    # Mapper grid search
    grid = CONFIG["mapper"]["grid_search"]
    keys, values = zip(*grid.items())
    mapper_rows = []
    mapper_summaries = {}
    short = {"num_cubes": "cubes", "perc_overlap": "overlap", "dbscan_eps": "eps"}

    for combination in product(*values):
        grid_params = dict(zip(keys, combination))
        mapper_params = {**CONFIG["mapper"]["fixed"], **grid_params}
        tag = "_".join(f"{k}-{_fmt(v)}" for k, v in grid_params.items())

        with timer.stage(f"Mapper subsample stability ({tag})"):
            _, summary = run_mapper_subsample_stability(
                feature_prefix=dev_prefix, csv_path=paths["master_csv"], name_col=cols["name"], split_tag="dev",
                mapper_params=mapper_params, random_state=seed, resample=CONFIG["mapper"]["resample"],
                plot_style=CONFIG["mapper"]["subsample_plot_style"],
                plot_path=out_dir / f"mapper_subsamples_{tag}.png", **plot_kw)
        mapper_summaries[", ".join(f"{short.get(k, k)}={_fmt(v)}" for k, v in grid_params.items())] = summary
        mapper_rows.append({**grid_params, **summary, "runtime_s": timer.rows[-1]["seconds"]})

        if CONFIG["mapper"]["write_html"]:
            run_mapper_pipeline(feature_prefix=dev_prefix, csv_path=paths["master_csv"], name_col=cols["name"],
                                mapper_params=mapper_params, random_state=seed,
                                output_html=out_dir / f"mapper_graph_{tag}.html")

    mapper_summary = pd.DataFrame(mapper_rows)
    mapper_summary.to_csv(out_dir / "mapper_grid_search.csv", index=False)
    print("\n=== Mapper grid search summary (ref = full-dev graph; mean / std across subsamples) ===")
    print(mapper_summary.to_string(index=False))

    if CONFIG["mapper"]["compare_grid"] and CONFIG["mapper"]["resample"]["n_draws"] > 1:
        plot_mapper_grid_comparison(mapper_summaries, styles=CONFIG["mapper"]["compare_styles"],
                                    plot_path=out_dir / "mapper_grid_comparison.png", **plot_kw)
    if CONFIG["mapper"]["resample"]["n_draws"] > 1:
        inter, heat = CONFIG["mapper"]["interaction_plot"], CONFIG["mapper"]["heatmap_plot"]
        labels = CONFIG["mapper"]["param_labels"]
        if inter["enabled"]:
            plot_mapper_grid_interaction(mapper_summary, x_param=inter["x_param"], color_param=inter["color_param"],
                                         fill_param=inter["fill_param"], color_styles=inter["color_styles"],
                                         param_labels=labels, x_offset=inter["x_offset"],
                                         plot_path=out_dir / "mapper_grid_interaction.png", **plot_kw)
        if heat["enabled"]:
            plot_mapper_grid_heatmap(mapper_summary, row_param=heat["row_param"], col_param=heat["col_param"],
                                     panel_param=heat["panel_param"], param_labels=labels, cmap=heat["cmap"],
                                     plot_path=out_dir / "mapper_grid_heatmap.png", **plot_kw)

    timer.save(out_dir / "runtime.csv")
    return fft_df, mapper_summary


if __name__ == "__main__":
    main()
