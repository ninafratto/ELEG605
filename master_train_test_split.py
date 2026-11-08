#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def create_master_splits(csv_path="labels.csv", output_path="master_dataset_split.csv", name_col ="labels", target_col ="continuous phases", seed=42):

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()
    df['split'] = 'dev'
    df['fold'] = -1
    
    # 80/20 stratified split dev/test
    dev_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=seed,
        stratify=df[target_col]
        )
    df.loc[test_idx, 'split'] = 'test'
    
    # 5-fold stratified cross-validation on dev set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    dev_df = df.loc[dev_idx]
    
    for fold_idx, (_, val_indices) in enumerate(skf.split(dev_df, dev_df[target_col])):
        original_indices = dev_df.index[val_indices]
        df.loc[original_indices, 'fold'] = fold_idx
    
    # 5. Save master file to disk
    df.to_csv(output_path, index=False)
    
    # Print summary verification
    print(f"Master split saved to '{output_path}'.")
    print("\ndataset split counts:")
    print(df['split'].value_counts())
    print("\ndevelopment set fold distribution:")
    print(pd.crosstab(df[df['split'] == 'dev'][target_col], df[df['split'] == 'dev']['fold']))

if __name__ == "__main__":
    create_master_splits()