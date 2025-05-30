# src/dataset.py

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict, Optional # <--- MODIFIED THIS LINE to include Optional

def apply_pca(X: np.ndarray, num_components: int) -> np.ndarray:
    """
    对高光谱数据应用主成分分析(PCA)进行降维。
    """
    h, w, bands = X.shape
    X_reshaped = np.reshape(X, (-1, bands))
    pca = PCA(n_components=num_components, whiten=True)
    X_pca = pca.fit_transform(X_reshaped)
    X_pca_reshaped = np.reshape(X_pca, (h, w, num_components))
    return X_pca_reshaped

def pad_with_zeros(X: np.ndarray, margin: int) -> np.ndarray:
    """
    在图像的边缘用0进行填充。
    """
    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    new_X[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return new_X

def create_image_cubes(X: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从高光谱数据中为每个像素提取3D图像块(cubes)。
    只提取带标签的像素（y > 0）。
    """
    margin = (window_size - 1) // 2
    zero_padded_X = pad_with_zeros(X, margin)
    labeled_pixels_coords = np.argwhere(y > 0)
    num_samples = len(labeled_pixels_coords)
    patches_data = np.zeros((num_samples, window_size, window_size, X.shape[2]))
    patches_labels = np.zeros(num_samples, dtype=int)

    for i, (r_orig, c_orig) in enumerate(labeled_pixels_coords):
        r_start_padded = r_orig
        c_start_padded = c_orig
        patch = zero_padded_X[r_start_padded : r_start_padded + window_size,
                              c_start_padded : c_start_padded + window_size]
        patches_data[i, :, :, :] = patch
        patches_labels[i] = y[r_orig, c_orig]
    patches_labels -= 1
    return patches_data, patches_labels

def create_dataloaders(
    X_all_cubes: np.ndarray,
    y_all_labels: np.ndarray,
    test_ratio: float,
    val_ratio: float, 
    batch_size: int,
    random_state: int,
    num_classes: int
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, np.ndarray]: # val_loader 可以是 Optional
    """
    将数据划分为训练、测试集，并创建对应的DataLoader。
    训练集的DataLoader将使用WeightedRandomSampler进行过采样。
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all_cubes, y_all_labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y_all_labels
    )

    val_loader: Optional[DataLoader] = None # Initialize val_loader as Optional DataLoader
    if val_ratio > 0 and len(y_train_val) > 0 : 
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_train_val
        )
    else: 
        X_train, y_train = X_train_val, y_train_val
        # Ensure X_val and y_val are suitable for to_pytorch_format if empty
        X_val, y_val = np.empty((0, X_all_cubes.shape[1], X_all_cubes.shape[2], X_all_cubes.shape[3]), dtype=X_all_cubes.dtype), np.array([], dtype=y_all_labels.dtype)


    train_class_counts = np.bincount(y_train, minlength=num_classes)
    
    sampler = None
    # Ensure y_train is not empty before accessing its elements for sampler weights
    if len(y_train) > 0 and np.any(train_class_counts > 0): 
        counts_for_sampler = np.array(train_class_counts, dtype=float)
        counts_for_sampler[counts_for_sampler == 0] = 1 
        class_weights_for_sampler = 1. / counts_for_sampler
        # Check if y_train is not empty before indexing class_weights_for_sampler
        if y_train.size > 0:
            sample_weights = class_weights_for_sampler[y_train]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else: # y_train is empty, cannot create sampler weights based on it
             sampler = None


    def to_pytorch_format(data_cubes: np.ndarray) -> np.ndarray:
        if data_cubes.ndim == 0 or data_cubes.shape[0] == 0:
            # Determine expected D from X_all_cubes if possible, or use a sensible default if not.
            # This part might need adjustment based on how X_all_cubes.shape[3] is guaranteed.
            # Assuming X_all_cubes.shape is (N, H, W, D_pca)
            d_pca = X_all_cubes.shape[3] if X_all_cubes.ndim == 4 and X_all_cubes.shape[0] > 0 else 0 # Default D if not inferable
            h_patch = X_all_cubes.shape[1] if X_all_cubes.ndim >= 2 and X_all_cubes.shape[0] > 0 else 0
            w_patch = X_all_cubes.shape[2] if X_all_cubes.ndim >= 3 and X_all_cubes.shape[0] > 0 else 0
            return np.empty((0, 1, d_pca, h_patch, w_patch), dtype=data_cubes.dtype)

        # (N, H, W, D, C_in=1) -> (N, C_in=1, D, H, W)
        # Original assumption for data_cubes: (num_samples, window_size, window_size, pca_components)
        reshaped_cubes = data_cubes.reshape(
            data_cubes.shape[0], data_cubes.shape[1], data_cubes.shape[2], data_cubes.shape[3], 1
        )
        return reshaped_cubes.transpose(0, 4, 3, 1, 2)

    X_train_tensor = torch.FloatTensor(to_pytorch_format(X_train))
    y_train_tensor = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler if len(y_train) > 0 else None, # Only use sampler if y_train is not empty
        shuffle=(sampler is None and len(y_train) > 0) # Shuffle if no sampler and y_train is not empty
    )

    if len(y_val) > 0: 
        X_val_tensor = torch.FloatTensor(to_pytorch_format(X_val))
        y_val_tensor = torch.LongTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # else val_loader remains None, as initialized

    X_test_tensor = torch.FloatTensor(to_pytorch_format(X_test))
    y_test_tensor = torch.LongTensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_class_counts