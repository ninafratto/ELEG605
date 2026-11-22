#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nefratto

Workflow module: physics-informed pipeline (FFT radial profiles -> L2 multinomial logistic regression).
All values (paths, columns, lambda, solver settings, plot options) are supplied by a conductor script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, adjusted_rand_score, confusion_matrix, classification_report
from structure_factor import load_and_preprocess_images, compute_2d_fft, radial_profile


def load_master_split_labels(*, csv_path, name_col, label_col):

    print("loading dataset...")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(int)
    label_map = dict(zip(df[name_col], df[label_col]))

    return label_map, df


def extract_dataset_features_by_split(*, images_dir, label_map, df_metadata, name_col, use_1d_radial):

    features_list = []
    names_list = []
    labels_list = []

    print("extracting features from image directory...")
    print("computing FFT...")

    for base_name, stacked_channels, phase_indices in load_and_preprocess_images(images_dir):

        power_spectra = compute_2d_fft(stacked_channels)

        if use_1d_radial:
            _, sq_matrix = radial_profile(power_spectra)
            feature_vector = sq_matrix.flatten()
        else:
            feature_vector = power_spectra.flatten()

        feature_vector = np.log10(feature_vector + 1e-6)
        features_list.append(feature_vector)
        names_list.append(base_name)
        labels_list.append(label_map[base_name])

    X_all = np.vstack(features_list)
    feature_df = pd.DataFrame({name_col: names_list})
    feature_df = feature_df.merge(df_metadata, on=name_col, how='left')

    return X_all, feature_df


def _draw_baseline(ax, baseline_accuracy, color):
    """Dashed horizontal line on the accuracy axis at the majority-class baseline (None -> nothing drawn)."""
    if baseline_accuracy is None:
        return None
    return ax.axhline(baseline_accuracy, color=color, linestyle='--', linewidth=1.5,
                      label=f'majority baseline ({100 * baseline_accuracy:.1f}%)')


def _errorbar(ax, x, y, yerr, *, marker, color, label=None, markersize=8):
    """Scatter-style points (no connecting line) with symmetric error bars."""
    return ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, ecolor=color, markersize=markersize,
                       linestyle='none', elinewidth=1.2, capsize=3, label=label)


def plot_cv_metrics(fold_losses, fold_accuracies, fold_loss_ses, fold_accuracy_ses, *, loss_color, accuracy_color,
                    baseline_accuracy, plot_path, show_plot, dpi):
    """Single lambda: per-fold cross-entropy (left axis) and validation accuracy (right axis), +/- within-fold SE."""

    print("plotting CV cross-entropy and validation accuracy...")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    folds = np.arange(1, len(fold_losses) + 1)

    # left y-axis: categorical cross-entropy (log loss); points nudged apart so the two sets of bars don't overlap
    ax1.set_xlabel('CV fold number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('categorical cross-entropy', color=loss_color, fontsize=11, fontweight='bold')
    _errorbar(ax1, folds - 0.05, fold_losses, fold_loss_ses, marker='o', color=loss_color)
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax1.set_xticks(folds)
    ax1.grid(True, alpha=0.5)

    # right y-axis: validation accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('validation accuracy', color=accuracy_color, fontsize=11, fontweight='bold')
    _errorbar(ax2, folds + 0.05, fold_accuracies, fold_accuracy_ses, marker='s', color=accuracy_color)
    ax2.tick_params(axis='y', labelcolor=accuracy_color)
    if _draw_baseline(ax2, baseline_accuracy, accuracy_color) is not None:
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    print(f"\nsaved CV evaluation plot to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_cv_lambda_comparison(cv_results, *, styles, baseline_accuracy, baseline_color, plot_path, show_plot, dpi):
    """
    Two subplots comparing lambdas per fold: (1) validation accuracy with the majority-class baseline,
    (2) categorical cross-entropy, each +/- its within-fold standard error. `cv_results` maps lambda -> the dict
    returned by train_and_evaluate_multinomial_logistic_regression; styles[i] = {"marker": ..., "color": ...}
    is used for the i-th lambda in both subplots. One shared legend.
    """
    print(f"plotting CV comparison for lambdas {list(cv_results)}...")

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5.5))
    offsets = np.linspace(-0.15, 0.15, len(cv_results)) if len(cv_results) > 1 else [0.0]

    for (l2_lambda, res), style, dx in zip(cv_results.items(), styles, offsets):
        folds = np.arange(1, len(res['fold_accuracies']) + 1)
        kw = dict(marker=style['marker'], color=style['color'], label=f'$\\lambda$ = {l2_lambda:g}')
        _errorbar(ax_acc, folds + dx, res['fold_accuracies'], res['fold_accuracy_ses'], **kw)
        _errorbar(ax_loss, folds + dx, res['fold_log_losses'], res['fold_log_loss_ses'], **kw)

    _draw_baseline(ax_acc, baseline_accuracy, baseline_color)

    for ax, ylabel in ((ax_acc, 'validation accuracy'), (ax_loss, 'categorical cross-entropy')):
        ax.set_xlabel('CV fold number', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticks(folds)
        ax.grid(True, alpha=0.5)

    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(plot_path, dpi=dpi)
    print(f"saved lambda comparison plot to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _make_pipeline(*, l2_lambda, solver, max_iter, random_state):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(solver=solver, max_iter=max_iter, C=1.0 / l2_lambda, random_state=random_state))


def train_and_evaluate_multinomial_logistic_regression(*, X_all, feature_df, label_col, l2_lambda, solver, max_iter,
                                                       random_state, loss_color, accuracy_color, baseline_accuracy,
                                                       plot_path, show_plot, dpi):
    """Development: K-fold CV over the dev folds. Returns a dict of mean and per-fold log loss / accuracy."""

    print("training multinomial logistic regression...")

    dev_mask = (feature_df['split'] == 'dev').values
    dev_df = feature_df[dev_mask].reset_index(drop=True)
    X_dev = X_all[dev_mask]
    y_dev = dev_df[label_col].values
    unique_classes = np.unique(y_dev)
    folds = sorted(dev_df['fold'].unique())

    print(f"total samples extracted: {len(feature_df)} | dev set: {len(y_dev)} | test set: {sum(feature_df['split'] == 'test')}")
    print(f"L2 lambda: {l2_lambda} -> C: {1.0 / l2_lambda:.6f}")

    val_accuracies = []
    val_log_losses = []
    val_accuracy_ses = []
    val_log_loss_ses = []

    for i, fold in enumerate(folds):

        train_fold_mask = (dev_df['fold'] != fold).values
        val_fold_mask = (dev_df['fold'] == fold).values
        X_train, y_train = X_dev[train_fold_mask], y_dev[train_fold_mask]
        X_val, y_val = X_dev[val_fold_mask], y_dev[val_fold_mask]

        pipeline = _make_pipeline(l2_lambda=l2_lambda, solver=solver, max_iter=max_iter, random_state=random_state)
        pipeline.fit(X_train, y_train)

        if len(y_val) > 0:

            y_val_pred = pipeline.predict(X_val)
            y_val_probs = pipeline.predict_proba(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            loss = log_loss(y_val, y_val_probs, labels=unique_classes)
            val_accuracies.append(acc)
            val_log_losses.append(loss)

            # within-fold standard errors: binomial SE for accuracy, SE of the per-sample cross-entropy terms
            n_val = len(y_val)
            p_true = y_val_probs[np.arange(n_val), np.searchsorted(pipeline.classes_, y_val)]
            sample_losses = -np.log(np.clip(p_true, np.finfo(float).eps, 1.0))
            val_accuracy_ses.append(float(np.sqrt(acc * (1 - acc) / n_val)))
            val_log_loss_ses.append(float(sample_losses.std(ddof=1) / np.sqrt(n_val)))

            print(f"fold {i + 1}/{len(folds)} | accuracy: {acc:.4f} +/- {val_accuracy_ses[-1]:.4f} (SE) | "
                  f"categorical cross-entropy: {loss:.4f} +/- {val_log_loss_ses[-1]:.4f} (SE)")

    print(f"\nmean {len(folds)}-fold CV accuracy: {np.mean(val_accuracies):.4f} +/- {np.std(val_accuracies):.4f}")
    print(f"mean {len(folds)}-fold CV cross-entropy: {np.mean(val_log_losses):.4f} +/- {np.std(val_log_losses):.4f}")

    plot_cv_metrics(val_log_losses, val_accuracies, val_log_loss_ses, val_accuracy_ses,
                    loss_color=loss_color, accuracy_color=accuracy_color,
                    baseline_accuracy=baseline_accuracy, plot_path=plot_path, show_plot=show_plot, dpi=dpi)

    return {
        "mean_log_loss": float(np.mean(val_log_losses)),
        "mean_accuracy": float(np.mean(val_accuracies)),
        "fold_log_losses": val_log_losses,
        "fold_accuracies": val_accuracies,
        "fold_log_loss_ses": val_log_loss_ses,
        "fold_accuracy_ses": val_accuracy_ses,
    }


def fit_multinomial_logistic_regression(*, X_all, feature_df, label_col, l2_lambda, solver, max_iter, random_state):
    """Final model: refits scaler + L2 logistic regression on the ENTIRE dev set (all folds). Test rows are never seen."""

    print(f"fitting final multinomial logistic regression on full dev set (L2 lambda = {l2_lambda})...")

    dev_mask = (feature_df['split'] == 'dev').values
    pipeline = _make_pipeline(l2_lambda=l2_lambda, solver=solver, max_iter=max_iter, random_state=random_state)
    pipeline.fit(X_all[dev_mask], feature_df.loc[dev_mask, label_col].values)

    print(f"trained on {dev_mask.sum()} dev samples.")
    return pipeline


def evaluate_multinomial_logistic_regression_on_test(*, model, X_all, feature_df, label_col, name_col):
    """Test: inference only with a pre-trained `model` pipeline on the held-out test split."""

    print("performing multinomial logistic inference...")
    print("evaluating pre-trained model on test set...")

    test_mask = (feature_df['split'] == 'test').values
    X_test = X_all[test_mask]
    y_test = feature_df.loc[test_mask, label_col].values

    # inference only
    y_test_pred = model.predict(X_test)
    y_test_probs = model.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_loss = log_loss(y_test, y_test_probs, labels=model.classes_)
    test_ari = adjusted_rand_score(y_test, y_test_pred)

    print(f"test set samples: {len(y_test)}")
    print(f"test accuracy: {test_acc:.4f}")
    print(f"test log loss: {test_loss:.4f}")
    print(f"test ARI: {test_ari:.4f}")
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, y_test_pred, labels=model.classes_),
                       index=model.classes_, columns=model.classes_))
    print("\n" + classification_report(y_test, y_test_pred, labels=model.classes_, zero_division=0))

    return {
        "method": "FFT + multinomial logistic regression",
        "n_test": int(len(y_test)),
        "test_accuracy": test_acc,
        "test_log_loss": test_loss,
        "test_ari": test_ari,
        "names": feature_df.loc[test_mask, name_col].values,
        "y_true": y_test,
        "y_pred": y_test_pred,
    }
