# train.py

import scipy.io as sio
import os
import types
import torch
import numpy as np
from typing import Any

from configs import config
from src.dataset import apply_pca, create_image_cubes, create_dataloaders
from src.model import HybridSN_BN
from src.trainer import Trainer
from src.utils import setup_logger

def main(config_module: types.ModuleType):
    cfg_dict = {k: v for k, v in vars(config_module).items() if k.isupper()}
    log_filename = f"{cfg_dict['DATASET_NAME']}_train_main_log.txt"
    logger = setup_logger(log_dir=cfg_dict['OUTPUT_PATH'], log_filename=log_filename)

    logger.info("----------- Project Configuration -----------")
    for key, value in cfg_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("---------------------------------------------")

    logger.info("Step 1: Loading data...")
    data_file = os.path.join(cfg_dict['DATA_PATH'], f"{cfg_dict['DATASET_NAME']}_corrected.mat")
    gt_file = os.path.join(cfg_dict['DATA_PATH'], f"{cfg_dict['DATASET_NAME']}_gt.mat")
    dataset_name_lower_for_key = cfg_dict["DATASET_NAME"].lower()
    X_mat_data = sio.loadmat(data_file)
    X = X_mat_data[f'{dataset_name_lower_for_key}_corrected']
    y_mat_data = sio.loadmat(gt_file)
    y = y_mat_data[f'{dataset_name_lower_for_key}_gt']
    logger.info(f"Original data shape: {X.shape}, Ground truth shape: {y.shape}")

    logger.info("Step 2: Preprocessing data...")
    if cfg_dict['USE_PCA']:
        X_pca = apply_pca(X, num_components=cfg_dict['PCA_COMPONENTS'])
    else:
        X_pca = X
    logger.info(f"Data shape after PCA (if applied): {X_pca.shape}")

    X_cubes, y_labels = create_image_cubes(X_pca, y, window_size=cfg_dict['PATCH_SIZE'])
    logger.info(f"Created {X_cubes.shape[0]} image cubes with shape {X_cubes.shape[1:]}")
    logger.info(f"Total labeled samples for splitting: {len(y_labels)}")

    logger.info("Step 3: Creating data loaders...")
    # create_dataloaders 现在返回 (train_loader, val_loader (Optional), test_loader, train_class_counts)
    train_loader, val_loader_maybe, test_loader, train_class_counts = create_dataloaders(
        X_cubes, y_labels,
        test_ratio=cfg_dict['TEST_RATIO'],
        val_ratio=cfg_dict['VALIDATION_RATIO'], # 设为0则 val_loader_maybe 为 None
        batch_size=cfg_dict['BATCH_SIZE'],
        random_state=cfg_dict['RANDOM_STATE'],
        num_classes=cfg_dict['NUM_CLASSES']
    )
    logger.info(f"Train DataLoader: {len(train_loader.dataset)} samples.")
    if val_loader_maybe:
        logger.info(f"Validation DataLoader: {len(val_loader_maybe.dataset)} samples.")
    else:
        logger.info("No validation set is used.")
    logger.info(f"Test DataLoader: {len(test_loader.dataset)} samples.")
    logger.info(f"Actual train class counts (before WeightedRandomSampler effect): {train_class_counts}")

    class_weights_tensor = None
    counts_for_weighting = np.array(train_class_counts, dtype=float)
    counts_for_weighting[counts_for_weighting == 0] = 1 # 处理0计数
    if np.any(train_class_counts == 0):
        logger.warning(f"Original train_class_counts had zeros: {train_class_counts}. Adjusted for weighting: {counts_for_weighting}.")
    
    if len(train_class_counts) > 0 and np.sum(counts_for_weighting) > 0 : # 确保有东西可以计算权重
        raw_weights = 1.0 / counts_for_weighting
        normalized_weights = raw_weights / np.sum(raw_weights)
        class_weights_tensor = torch.FloatTensor(normalized_weights)
        logger.info(f"Calculated class weights for loss function: {class_weights_tensor.tolist()}")
    else:
        logger.warning("Cannot calculate class weights due to empty or all-zero train_class_counts.")


    logger.info("Step 4: Initializing model...")
    model = HybridSN_BN(in_channels=cfg_dict['INPUT_CHANNELS'], out_channels=cfg_dict['NUM_CLASSES'])

    logger.info("Step 5: Initializing trainer and starting training...")
    device = torch.device(cfg_dict['DEVICE'])
    trainer = Trainer(model, cfg_dict, device, class_weights=class_weights_tensor)
    trainer.train(train_loader) # 只传递 train_loader

    logger.info("----------- Final Evaluation on Test Set -----------")
    test_acc, _ = trainer.evaluate(test_loader)

if __name__ == '__main__':
    main(config)