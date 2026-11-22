#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conductor: repeated dev/test splits with nested hyperparameter selection (post hoc robustness check).

For every seed s:
  1. new stratified dev/test split (split seed = s; fold / K-means / Mapper seed = s)
  2. full conductor_dev workflow on that split's dev set (same grids as conductor_dev.CONFIG)
  3. hyperparameters chosen AUTOMATICALLY from that split's dev results with the fixed rules in CONFIG["selection"]
  4. full conductor_test workflow on that split's test set with the selected hyperparameters
The test set of a split is never used to choose anything. Every other value comes from conductor_dev / conductor_test.

MoCo features do not depend on the split, so they are computed once (from the existing dev + test caches) and
re-partitioned per split; the network is not run again.

Outputs:
  results_repeated_split/seed_<s>/dev/, seed_<s>/test/     full dev + test outputs for split s
  results_repeated_split/selections.csv                    hyperparameters selected per split
  results_repeated_split/{fft,kmeans,mapper}_per_seed.csv  test metrics per split
  results_repeated_split/test_metric_heatmap.png, fft_seed_summary.png, mapper_seed_summary.png
  results_repeated_split/confusion_matrices.png/.xlsx       FFT, K-means, Mapper; counts pooled over all splits
  results_repeated_split/seed_summary.xlsx                 sheets Selections, Selection counts, FFT, K-means, Mapper, Seeds
  results_repeated_split/test_runtime_per_split.csv        test wall time per pipeline per split (also sheet Runtime (test))
  results_repeated_split/runtime.csv                       full dev + test wall time per split
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from conductor_dev import CONFIG as DEV_CONFIG, run_dev
from conductor_dev_seeds import CONFIG as DEV_SEEDS_CONFIG
from conductor_test import CONFIG as TEST_CONFIG, run_test
from conductor_test_seeds import CONFIG as TEST_SEEDS_CONFIG
from master_train_test_split import create_master_splits
from seed_summary import (select_by_rank_sum, summarize_metrics_long, write_sheets_excel, plot_metrics_per_seed,
                          plot_method_metric_heatmap, pooled_confusion, plot_confusion_matrices)
from runtime_log import RuntimeLog


CONFIG = {
    "seeds": DEV_SEEDS_CONFIG["seeds"],          # split s uses seed s for the split AND for the folds / algorithms
    "output_root": "./results_repeated_split",
    # existing MoCo caches; together they must cover every image in labels.csv
    "feature_sources": [f"{DEV_CONFIG['paths']['feature_prefix']}_dev", f"{DEV_CONFIG['paths']['feature_prefix']}_test"],
    # FIXED selection rules (set before running). Each criterion ranks the grid (1 = best); lowest summed rank
    # wins; ties broken by the criteria in the order listed.
    "selection": {
        "fft": [("cv_accuracy", "max"), ("cv_log_loss", "min")],
        "mapper": [("mean_stability", "max"), ("mean_connectedness", "max")],
    },
    "mapper_grid_keys": list(DEV_CONFIG["mapper"]["grid_search"]),
    # the hyperparameters chosen on the original split (for the agreement count)
    "original": {"l2_lambda": TEST_CONFIG["fft"]["l2_lambda"],
                 **{k: TEST_CONFIG["mapper"]["params"][k] for k in DEV_CONFIG["mapper"]["grid_search"]}},
    "write_html": False,                          # per-grid-combination dev graphs (8 per split) are not needed here
    "excel_decimals": 3,
    "heatmap_show_std": True,                     # write ± std across splits under each mean in test_metric_heatmap
    # test-stage wall time per pipeline = sum of these conductor_test runtime stages. MoCo extraction is not listed:
    # features are cached and re-partitioned per split, so it never runs here.
    "runtime_pipelines": {
        "FFT": ["FFT feature extraction", "FFT logreg fit + test"],
        "K-means": ["K-means fit + test"],
        "Mapper": ["Mapper fit + test"],
        "Mapper subsample stability": ["Mapper test subsample stability"],
    },
    # True -> skip the dev stage and re-run only the test stage with the hyperparameters in selections.csv
    "rerun_tests_only": False,
    # confusion matrices pooled over all splits (columns of seed_<s>/test/test_predictions.csv)
    "confusion": {
        "panels": [("FFT", "fft_logreg"), ("K-means", "kmeans_vote"), ("Mapper", "mapper_vote")],
        "true_col": "true",
        "labels": [0, 1, 2],                      # continuous-phase classes
        "extra_pred": {-1: "unassigned"},         # Mapper test samples in no dev-labelled node
        "cmap": "Blues",
        "xlabel": "predicted continuous phases",
        "ylabel": "true continuous phases",
    },
}


def build_feature_cache(sources):
    """Concatenate the per-split MoCo caches into one all-image cache (features are per image, not per split)."""
    X, y, names = [], [], []
    for p in sources:
        X.append(np.load(f"{p}_X.npy"))
        y.append(np.load(f"{p}_y.npy"))
        names.append(np.load(f"{p}_names.npy"))
    X, y, names = np.concatenate(X), np.concatenate(y), np.concatenate(names)
    if len(set(map(str, names))) != len(names):
        raise ValueError("feature sources overlap: an image appears in more than one cache")
    return X, y, names


def write_split_features(cache, *, master_csv, name_col, prefix):
    """Write <prefix>_{dev,test}_{X,y,names}.npy holding exactly the images of each split."""
    X, y, names = cache
    idx = {str(n): i for i, n in enumerate(names)}
    df = pd.read_csv(master_csv)
    df[name_col] = df[name_col].astype(str).str.strip()
    missing = sorted(set(df[name_col]) - set(idx))
    if missing:
        raise ValueError(f"{len(missing)} images have no cached MoCo features, e.g. {missing[:3]}")
    for tag in ("dev", "test"):
        rows = [idx[n] for n in df.loc[df["split"] == tag, name_col]]
        np.save(f"{prefix}_{tag}_X.npy", X[rows])
        np.save(f"{prefix}_{tag}_y.npy", y[rows])
        np.save(f"{prefix}_{tag}_names.npy", names[rows])


def main():
    root = Path(CONFIG["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    cols = DEV_CONFIG["columns"]
    cache = build_feature_cache(CONFIG["feature_sources"])
    timer = RuntimeLog()

    selections, runs = [], []
    for seed in CONFIG["seeds"]:
        print(f"\n################ split {seed} ################")
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        master_csv = str(seed_dir / "master_dataset_split.csv")
        prefix = str(seed_dir / DEV_CONFIG["paths"]["feature_prefix"])

        dev_cfg = copy.deepcopy(DEV_CONFIG)
        dev_cfg["random_seed"] = seed
        dev_cfg["split"]["seed"] = seed
        dev_cfg["paths"].update(master_csv=master_csv, feature_prefix=prefix, output_dir=str(seed_dir / "dev"))
        dev_cfg["mapper"]["write_html"] = CONFIG["write_html"]

        # the split is deterministic, so creating it here and again inside run_dev gives the same file
        create_master_splits(csv_path=dev_cfg["paths"]["labels_csv"], output_path=master_csv, name_col=cols["name"],
                             target_col=cols["target"], test_size=dev_cfg["split"]["test_size"],
                             n_folds=dev_cfg["split"]["n_folds"], seed=seed, fold_seed=seed)
        write_split_features(cache, master_csv=master_csv, name_col=cols["name"], prefix=prefix)

        with timer.stage(f"split {seed}"):
            fft_df, mapper_df = run_dev(dev_cfg)

            fft_pick, fft_ranked = select_by_rank_sum(fft_df, criteria=CONFIG["selection"]["fft"])
            mp_pick, mp_ranked = select_by_rank_sum(mapper_df, criteria=CONFIG["selection"]["mapper"])
            fft_ranked.to_csv(seed_dir / "fft_selection.csv", index=False)
            mp_ranked.to_csv(seed_dir / "mapper_selection.csv", index=False)
            sel = {"seed": seed, "l2_lambda": float(fft_pick["l2_lambda"]),
                   **{k: mp_pick[k] for k in CONFIG["mapper_grid_keys"]}}
            sel["fft_matches_original"] = bool(np.isclose(sel["l2_lambda"], CONFIG["original"]["l2_lambda"]))
            sel["mapper_matches_original"] = all(np.isclose(float(sel[k]), float(CONFIG["original"][k]))
                                                 for k in CONFIG["mapper_grid_keys"])
            selections.append(sel)
            print(f"\n[SELECTED] split {seed}: {sel}")

            runs.append(run_test(_test_cfg(seed_dir, seed, sel)).assign(seed=seed))

    sel_df = pd.DataFrame(selections)
    sel_df.to_csv(root / "selections.csv", index=False)
    _summarize(root, sel_df, pd.concat(runs, ignore_index=True), timer, "runtime.csv")


def _test_cfg(seed_dir, seed, sel):
    """conductor_test config for one split with that split's selected hyperparameters."""
    cfg = copy.deepcopy(TEST_CONFIG)
    cfg["random_seed"] = seed
    cfg["paths"].update(master_csv=str(seed_dir / "master_dataset_split.csv"),
                        feature_prefix=str(seed_dir / DEV_CONFIG["paths"]["feature_prefix"]),
                        output_dir=str(seed_dir / "test"))
    cfg["fft"]["l2_lambda"] = float(sel["l2_lambda"])
    for k in CONFIG["mapper_grid_keys"]:
        cfg["mapper"]["params"][k] = type(TEST_CONFIG["mapper"]["params"][k])(sel[k])
    return cfg


def rerun_tests():
    """Test stage only, for every split in selections.csv (dev results and selections are left untouched)."""
    root = Path(CONFIG["output_root"])
    sel_df = pd.read_csv(root / "selections.csv")
    timer, runs = RuntimeLog(), []
    for _, sel in sel_df.iterrows():
        seed = int(sel["seed"])
        print(f"\n################ split {seed} (test only) ################")
        with timer.stage(f"split {seed}"):
            runs.append(run_test(_test_cfg(root / f"seed_{seed}", seed, sel)).assign(seed=seed))
    # separate file so the full dev + test runtime log (runtime.csv) is not overwritten
    _summarize(root, sel_df, pd.concat(runs, ignore_index=True), timer, "runtime_test_rerun.csv")


def _stage_seconds(path, stage):
    rt = pd.read_csv(path)
    return float(rt.loc[rt["stage"] == stage, "seconds"].sum())


def test_runtime_summary(root):
    """
    Test-stage wall time per pipeline for every split (from seed_<s>/test/runtime.csv), grouped by
    CONFIG["runtime_pipelines"]. Returns (one row per split, one row per pipeline with mean ± std across splits).
    """
    rows = []
    for s in CONFIG["seeds"]:
        rt = pd.read_csv(Path(root) / f"seed_{s}" / "test" / "runtime.csv")
        row = {"seed": s}
        for pipe, stages in CONFIG["runtime_pipelines"].items():
            row[pipe] = float(rt.loc[rt["stage"].isin(stages), "seconds"].sum())
        rows.append(row)
    per_split = pd.DataFrame(rows)
    d = CONFIG["excel_decimals"]
    summary = pd.DataFrame([{"pipeline": p, "stages": " + ".join(st), "n_splits": len(per_split),
                             "mean_s": per_split[p].mean(), "std_s": per_split[p].std(),
                             "mean ± std (s)": f"{per_split[p].mean():.{d}f} ± {per_split[p].std():.{d}f}"}
                            for p, st in CONFIG["runtime_pipelines"].items()])
    return per_split, summary


def _summarize(root, sel_df, per_seed, timer, runtime_file):

    sheets = {"Selections": sel_df}
    counts = []
    for name, keys in (("FFT", ["l2_lambda"]), ("Mapper", CONFIG["mapper_grid_keys"])):
        c = sel_df.groupby(keys).size().rename("n_splits").reset_index()
        counts.append(c.assign(pipeline=name)[["pipeline", *keys, "n_splits"]])
    sheets["Selection counts"] = pd.concat(counts, ignore_index=True)

    per_method = {}
    for sheet, (file_key, method, values) in TEST_SEEDS_CONFIG["methods"].items():
        rows = per_seed[per_seed["method"] == method]
        per_method[sheet] = rows[["seed", *values]].reset_index(drop=True)
        per_method[sheet].to_csv(root / f"{file_key}_per_seed.csv", index=False)
        sheets[sheet] = summarize_metrics_long(rows, value_cols=values, value_labels=TEST_SEEDS_CONFIG["value_labels"],
                                               decimals=CONFIG["excel_decimals"])
        print(f"\n=== {sheet} across splits ===")
        print(sheets[sheet].to_string(index=False))
    runtime_per_split, sheets["Runtime (test)"] = test_runtime_summary(root)
    runtime_per_split.to_csv(root / "test_runtime_per_split.csv", index=False)
    print("\n=== Test runtime per pipeline (s) across splits ===")
    print(sheets["Runtime (test)"].to_string(index=False))
    sheets["Seeds"] = pd.DataFrame({
        "seed": CONFIG["seeds"], "output_folder": [f"seed_{s}" for s in CONFIG["seeds"]],
        "dev_runtime_s": [_stage_seconds(root / f"seed_{s}" / "dev" / "runtime.csv", "TOTAL") for s in CONFIG["seeds"]],
        "test_runtime_s": [_stage_seconds(root / f"seed_{s}" / "test" / "runtime.csv", "TOTAL") for s in CONFIG["seeds"]]})
    write_sheets_excel(path=root / "seed_summary.xlsx", sheets=sheets)
    print("\n=== Selections ===")
    print(sel_df.to_string(index=False))

    make_summary_figures(root, per_method)
    timer.save(root / runtime_file)


def make_summary_figures(root, per_method=None):
    """Same figures as results_test_seed (styles / panels from conductor_test_seeds), one point per split."""
    root = Path(root)
    tcfg = TEST_SEEDS_CONFIG
    if per_method is None:
        per_method = {sheet: pd.read_csv(root / f"{key}_per_seed.csv") for sheet, (key, _, _) in tcfg["methods"].items()}
    plot_kw = {"show_plot": TEST_CONFIG["plots"]["show"], "dpi": TEST_CONFIG["plots"]["dpi"]}
    baselines = {} if tcfg["baseline_accuracy"] is None else {
        "test_accuracy": (tcfg["baseline_accuracy"], tcfg["baseline_color"],
                          f"majority baseline ({tcfg['baseline_accuracy']:.1%})")}
    plot_metrics_per_seed(per_method["FFT"], panels=tcfg["fft_panels"], style=tcfg["fft_style"],
                          baselines=baselines, plot_path=root / "fft_seed_summary.png", **plot_kw)
    plot_metrics_per_seed(per_method["Mapper"], panels=tcfg["mapper_panels"], style=tcfg["mapper_style"],
                          baselines={}, plot_path=root / "mapper_seed_summary.png", **plot_kw)
    hm = tcfg["heatmap"]
    if hm["enabled"]:
        plot_method_metric_heatmap(per_method, methods=list(tcfg["methods"]), metrics=hm["metrics"],
                                   metric_labels=hm["metric_labels"], cmap=hm["cmap"], vmin=hm["vmin"],
                                   vmax=hm["vmax"], annotate=hm["annotate"], decimals=CONFIG["excel_decimals"],
                                   na_color=hm["na_color"], na_text_color=hm["na_text_color"],
                                   plot_path=root / "test_metric_heatmap.png", show_std=CONFIG["heatmap_show_std"],
                                   **plot_kw)

    cc = CONFIG["confusion"]
    preds = pd.concat([pd.read_csv(root / f"seed_{s}" / "test" / "test_predictions.csv").assign(seed=s)
                       for s in CONFIG["seeds"]], ignore_index=True)
    matrices = [(title, pooled_confusion(preds, true_col=cc["true_col"], pred_col=col, labels=cc["labels"],
                                         extra_pred=cc["extra_pred"])) for title, col in cc["panels"]]
    with pd.ExcelWriter(root / "confusion_matrices.xlsx", engine="openpyxl") as xw:
        for title, cm in matrices:
            cm.rename_axis(index="true \\ predicted").to_excel(xw, sheet_name=title)
    plot_confusion_matrices(matrices, cmap=cc["cmap"], xlabel=cc["xlabel"], ylabel=cc["ylabel"],
                            plot_path=root / "confusion_matrices.png", **plot_kw)


if __name__ == "__main__":
    rerun_tests() if CONFIG["rerun_tests_only"] else main()
