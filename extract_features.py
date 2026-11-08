#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:09:58 2026

@author: nefratto
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Import exact functions from structure_factor.py
from structure_factor import load_and_preprocess_images


class MoCoStructureFactorDataset(Dataset):

    def __init__(self, images_dir="./Images_set", split_csv="master_dataset_split.csv", 
                 split_tag="dev", name_col="labels", label_col="continuous phases"):
        
        self.split_tag = split_tag
        df = pd.read_csv(split_csv)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        df[name_col] = df[name_col].astype(str).str.strip()
        
        # Filter dataframe strictly by split_tag ('dev' vs 'test')
        split_df = df[df['split'] == split_tag].copy()
        self.label_map = dict(zip(split_df[name_col], split_df[label_col]))
        
        self.samples = []
        self.labels = []
        self.base_names = []
        
        print(f"Loading image set strictly for split '{split_tag}'...")
        for base_name, stacked_channels, _ in load_and_preprocess_images(images_dir):
            if base_name in self.label_map:
                img_tensor = torch.from_numpy(stacked_channels).float()
                
                # Spatial standardization across spatial dims
                mean = img_tensor.mean(dim=(-2, -1), keepdim=True)
                std = img_tensor.std(dim=(-2, -1), keepdim=True) + 1e-6
                img_tensor = (img_tensor - mean) / std
                
                self.samples.append(img_tensor)
                self.labels.append(self.label_map[base_name])
                self.base_names.append(base_name)
                
        print(f"--> Loaded {len(self.samples)} samples for split '{split_tag}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.base_names[idx]


def load_moco_resnet50(weights_path="moco_v2_800ep_pretrain.pth.tar", in_channels=3):

    model = models.resnet50(weights=None)
    
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Identity()

    if Path(weights_path).exists():
        print(f"Loading Meta MoCo-v2 ResNet-50 weights from '{weights_path}'...")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        cleaned_state_dict = {}
        
        for k, v in state_dict.items():
            k_clean = k.replace("module.encoder_q.", "").replace("encoder_q.", "").replace("module.", "")
            if not k_clean.startswith("fc"):
                cleaned_state_dict[k_clean] = v
                
        msg = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Successfully loaded backbone weights! (Missing keys ignored: {len(msg.missing_keys)})")
    else:
        print(f"[NOTICE] Weight file '{weights_path}' not found. Using untrained ResNet-50 backbone for feature extraction.")

    return model, num_ftrs


def extract_and_save_features(model, dataloader, device, output_prefix="moco_features_dev"):
    model.eval()
    features_list = []
    labels_list = []
    names_list = []

    with torch.no_grad():
        for imgs, labels, names in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            
            features_list.append(feats.cpu().numpy())
            labels_list.extend(labels.numpy())
            names_list.extend(names)

    if features_list:
        X_feats = np.vstack(features_list)
        y_labels = np.array(labels_list)
        names_arr = np.array(names_list)

        np.save(f"{output_prefix}_X.npy", X_feats)
        np.save(f"{output_prefix}_y.npy", y_labels)
        np.save(f"{output_prefix}_names.npy", names_arr)
        
        print(f"Saved extracted features: {output_prefix}_X.npy shape {X_feats.shape} | Labels: {y_labels.shape}")
        return X_feats, y_labels
    else:
        print(f"[WARNING] No features extracted for {output_prefix}.")
        return None, None


def extract_features_for_split(split_tag="dev"):
    """Entry point for running feature extraction on a specific split tag."""
    images_directory = "./Images_set"
    master_csv = "./master_dataset_split.csv"
    moco_weights_file = "./moco_v2_800ep_pretrain.pth.tar"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    model, feature_dim = load_moco_resnet50(
        weights_path=moco_weights_file, 
        in_channels=3
    )
    model = model.to(device)

    dataset = MoCoStructureFactorDataset(
        images_dir=images_directory,
        split_csv=master_csv,
        split_tag=split_tag
    )
    
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        extract_and_save_features(
            model=model, 
            dataloader=loader, 
            device=device, 
            output_prefix=f"moco_features_{split_tag}"
        )


if __name__ == "__main__":
    # Default execution extracts features strictly for dev split
    extract_features_for_split(split_tag="dev")