#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow module: stratified dev/test split + stratified K-fold assignment on the dev set.

All values (paths, column names, split fractions, seeds) are supplied by a conductor script.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def create_master_splits(*, csv_path, output_path, name_col, target_col, test_size, n_folds, seed, fold_seed):
    """seed sets the dev/test split; fold_seed sets the K-fold assignment within dev (can vary with the test set fixed)."""

    print("performing development/test split...")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()
    df['split'] = 'dev'
    df['fold'] = -1

    # stratified split dev/test
    dev_idx, test_idx = train_test_split(df.index, test_size=test_size, random_state=seed, stratify=df[target_col])
    df.loc[test_idx, 'split'] = 'test'

    # stratified K-fold cross-validation on dev set
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
    dev_df = df.loc[dev_idx]

    for fold_idx, (_, val_indices) in enumerate(skf.split(dev_df, dev_df[target_col])):
        original_indices = dev_df.index[val_indices]
        df.loc[original_indices, 'fold'] = fold_idx

    # save master file to disk
    df.to_csv(output_path, index=False)

    # print summary verification
    print(f"master split saved to '{output_path}'")
    print("\ndataset split counts:")
    print(df['split'].value_counts())
    print("\ndevelopment set fold distribution:")
    print(pd.crosstab(df[df['split'] == 'dev'][target_col], df[df['split'] == 'dev']['fold']))

    return df
