#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:03:18 2026

@author: neens
"""

from pathlib import Path
from collections import defaultdict
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_images(folder_path="./Images_set"):

    folder = Path(folder_path)
    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    groups = defaultdict(dict) 

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in valid_extensions:
            match = re.match(r"^(image_\d+)_(\d+)$", file_path.stem)
            if match:
                base_name = match.group(1)      # image_0002
                phase_idx = int(match.group(2)) # 0, 1, or 2
                #print(f"[DEBUG] File: {file_path.name} -> base_name: '{base_name}', phase_idx: {phase_idx}")
                groups[base_name][phase_idx] = file_path
            #else:
                #print(f"[DEBUG FAILED REGEX] File: {file_path.name} (stem: '{file_path.stem}')")

    for base_name, phase_dict in groups.items():
        sorted_indices = sorted(phase_dict.keys())
        channel_data_list = []
        
        for idx in sorted_indices:
            file_path = phase_dict[idx]
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            data = img.astype(np.float64)
            data -= np.mean(data)
            channel_data_list.append(data)
        
        if channel_data_list:
            stacked_data = np.stack(channel_data_list, axis=0)
            yield base_name, stacked_data, sorted_indices

def compute_2d_fft(data):
    
    fft_2d = np.fft.fft2(data, axes=(-2,-1))
    fft_shifted = np.fft.fftshift(fft_2d, axes=(-2,-1))
    return np.abs(fft_shifted)**2

def radial_profile(power_spectrum):
    
    if power_spectrum.ndim == 3:
        profiles = []
        for ps in power_spectrum:
            q, sq = radial_profile(ps)
            profiles.append(sq)
        return q, np.array(profiles)
    
    ny, nx = power_spectrum.shape
    cy, cx = ny // 2, nx // 2
    
    y, x = np.ogrid[-cy:ny-cy, -cx:nx-cx]
    r = np.hypot(x, y).astype(int)
    
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    
    nonzero = nr > 0
    sq = np.zeros_like(tbin, dtype=float)
    sq[nonzero] = tbin[nonzero] / nr[nonzero]
    
    max_q = min(cy, cx)
    return np.arange(max_q), sq[:max_q]

def main():
    images_dir = "./Images_set"  # path to your images folder
    plt.figure(figsize=(10, 6))
    
    for base_name, stacked_channels, phase_indices in load_and_preprocess_images(images_dir):
        power_spectra = compute_2d_fft(stacked_channels) # 2D FFT
        q, sq_matrix = radial_profile(power_spectra) # 1D radial average
        
        for i, phase_idx in enumerate(phase_indices):
            sq = sq_matrix[i]
            label_name = f"{base_name} | channel {phase_idx}"
            plt.plot(q, sq, label=label_name)
            
            if len(sq) > 1:
                q_peak = q[1:][np.argmax(sq[1:])] + 1
                domain_size = 2 * np.pi / q_peak
                print(f"sample: {base_name:15s} | channel {phase_idx:d} | peak q: {q_peak:3d} px^-1 | domain size ~ {domain_size:.2f} px")

    plt.xlabel(r"wavenumber, $q$, (pixels$^{-1}$)")
    plt.ylabel(r"structure factor, $S(q)$")
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed_time = time.perf_counter() - start_time
    print(f"\n[INFO] Execution time: {elapsed_time:.2f} seconds.")
