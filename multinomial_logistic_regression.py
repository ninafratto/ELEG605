#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 08:11:06 2026

@author: nefratto
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss

# Import exact functions from structure_factor.py
from structure_factor import load_and_preprocess_images, compute_2d_fft, radial_profile


def load_master_split_labels(csv_path="master_dataset_split.csv", name_col="labels", label_col="continuous phases"):
    """
    Loads master dataset split mapping containing sample base names, continuous labels,
    split tags ('dev' vs 'test'), and cross-validation fold IDs (0-4).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(int)
    
    label_map = dict(zip(df[name_col], df[label_col]))
    return label_map, df


def extract_dataset_features_by_split(images_dir="./Images_set", label_map=None, df_metadata=None, name_col="labels", use_1d_radial=True):
    """
    Processes images via structure_factor.py and extracts features for all samples,
    returning aligned arrays and metadata.
    """
    features_list = []
    names_list = []
    labels_list = []

    print("Extracting features from image directory...")
    for base_name, stacked_channels, phase_indices in load_and_preprocess_images(images_dir):
        if label_map is not None and base_name not in label_map:
            print(f"[WARNING] '{base_name}' found in images folder but missing from CSV labels. Skipping.")
            continue
        
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


def plot_cv_metrics(fold_losses, fold_accuracies):
    """Generates plots for 5-Fold CV Log Loss and Accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    folds = np.arange(1, len(fold_losses) + 1)
    
    color = 'tab:red'
    axes[0].set_xlabel('CV fold number', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('categorical cross-entropy', color=color, fontsize=11, fontweight='bold')
    axes[0].scatter(folds, fold_losses, color=color, marker='o', linewidth=2, label='Val Log Loss')
    axes[0].tick_params(axis='y', labelcolor=color)
    axes[0].set_xticks(folds)
    axes[0].grid(True, alpha=0.5)

    ax2 = axes[0].twinx()
    color = 'tab:blue'
    ax2.set_ylabel('validation accuracy', color=color, fontsize=11, fontweight='bold')
    ax2.scatter(folds, fold_accuracies, color=color, marker='s', linewidth=2, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    axes[1].bar(folds, fold_losses, color='crimson', alpha=0.7, edgecolor='black')
    axes[1].axhline(np.mean(fold_losses), color='black', linestyle='--', label=f'Mean Loss: {np.mean(fold_losses):.4f}')
    axes[1].set_xlabel('CV fold number', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('categorical cross-entropy', fontsize=11, fontweight='bold')
    axes[1].set_xticks(folds)
    axes[1].set_title('Log Loss Distribution across CV Folds', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("cv_log_loss_metrics.png", dpi=300)
    print("\nSaved CV evaluation plot to 'cv_log_loss_metrics.png'")
    plt.show()


def train_and_evaluate_multinomial_logistic_regression(X_all, feature_df, label_col="continuous phases", l2_lambda=0.1, random_state=42):
    """
    Fits Multinomial Logistic Regression using 5-fold cross-validation strictly on the Development set.
    HELD-OUT TEST SET IS NEVER ACCESSED.
    """
    # Filter strictly for development set
    dev_mask = (feature_df['split'] == 'dev').values
    dev_df = feature_df[dev_mask].reset_index(drop=True)
    
    X_dev = X_all[dev_mask]
    y_dev = dev_df[label_col].values
    
    unique_classes = np.unique(y_dev)
    C_val = 1.0 / l2_lambda

    print(f"\n================ CONFIGURATION (DEVELOPMENT ONLY) ================")
    print(f"Total Samples Extracted: {len(feature_df)} | Dev Set (In Use): {len(y_dev)} | Test Set (Isolated): {sum(feature_df['split'] == 'test')}")
    print(f"L2 lambda: {l2_lambda} -> converted C parameter: {C_val:.6f}")

    # 5-Fold Cross-Validation Loop over Development Set
    print("\n================ 5-FOLD CROSS-VALIDATION ================")
    val_accuracies = []
    val_log_losses = []
    
    for fold in range(5):
        train_fold_mask = (dev_df['fold'] != fold).values
        val_fold_mask = (dev_df['fold'] == fold).values
        
        X_train, y_train = X_dev[train_fold_mask], y_dev[train_fold_mask]
        X_val, y_val = X_dev[val_fold_mask], y_dev[val_fold_mask]

        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                C=C_val,
                random_state=random_state
            )
        )

        pipeline.fit(X_train, y_train)
        
        if len(y_val) > 0:
            y_val_pred = pipeline.predict(X_val)
            y_val_probs = pipeline.predict_proba(X_val)
            
            acc = accuracy_score(y_val, y_val_pred)
            loss = log_loss(y_val, y_val_probs, labels=unique_classes)
            
            val_accuracies.append(acc)
            val_log_losses.append(loss)
            
            print(f"Fold {fold + 1}/5 | Accuracy: {acc:.4f} | Categorical Cross-Entropy (Log Loss): {loss:.4f}")

    print(f"\nMean 5-Fold CV Accuracy : {np.mean(val_accuracies):.4f} +/- {np.std(val_accuracies):.4f}")
    print(f"Mean 5-Fold CV Log Loss : {np.mean(val_log_losses):.4f} +/- {np.std(val_log_losses):.4f}")

    plot_cv_metrics(val_log_losses, val_accuracies)
    return np.mean(val_log_losses), np.mean(val_accuracies)


if __name__ == "__main__":
    images_directory = "./Images_set"
    csv_labels_path = "./master_dataset_split.csv"
    name_column = "labels"
    label_column = "continuous phases"
    
    labels_dict, df_metadata = load_master_split_labels(
        csv_path=csv_labels_path, 
        name_col=name_column, 
        label_col=label_column
    )
    
    X_all, feature_df = extract_dataset_features_by_split(
        images_dir=images_directory,
        label_map=labels_dict,
        df_metadata=df_metadata,
        name_col=name_column,
        use_1d_radial=True
    )
    
    if len(feature_df) > 0:
        mean_loss, mean_acc = train_and_evaluate_multinomial_logistic_regression(
            X_all=X_all,
            feature_df=feature_df,
            label_col=label_column,
            l2_lambda=0.1
        )