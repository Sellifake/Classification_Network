# predict.py

import torch
import scipy.io as sio
import numpy as np
import os
import types
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from configs import config
from src.model import HybridSN_BN
from src.dataset import apply_pca, pad_with_zeros
from src.utils import setup_logger, plot_classification_map # 确保导入 plot_classification_map

def predict_full_map(config_module: types.ModuleType):
    cfg_dict = {k: v for k, v in vars(config_module).items() if k.isupper()}
    log_filename = f"{cfg_dict['DATASET_NAME']}_predict_main_log.txt"
    logger = setup_logger(log_dir=cfg_dict['OUTPUT_PATH'], log_filename=log_filename)
    device = torch.device(cfg_dict['DEVICE'])

    logger.info("----------- Prediction Configuration -----------")
    for key, value in cfg_dict.items():
        if key.isupper():
            logger.info(f"{key}: {value}")
    logger.info("------------------------------------------------")

    logger.info("Step 1: Loading full dataset for prediction...")
    data_file = os.path.join(cfg_dict['DATA_PATH'], f"{cfg_dict['DATASET_NAME']}_corrected.mat")
    gt_file = os.path.join(cfg_dict['DATA_PATH'], f"{cfg_dict['DATASET_NAME']}_gt.mat")
    dataset_name_lower_for_key = cfg_dict["DATASET_NAME"].lower()
    X_mat_data = sio.loadmat(data_file)
    X_original = X_mat_data[f'{dataset_name_lower_for_key}_corrected']
    y_mat_data = sio.loadmat(gt_file)
    y_ground_truth = y_mat_data[f'{dataset_name_lower_for_key}_gt']
    logger.info(f"Original data shape: {X_original.shape}, Ground truth shape: {y_ground_truth.shape}")

    logger.info("Step 2: Preprocessing data...")
    if cfg_dict['USE_PCA']:
        X_processed = apply_pca(X_original, num_components=cfg_dict['PCA_COMPONENTS'])
    else:
        X_processed = X_original
    logger.info(f"Data shape after PCA (if applied): {X_processed.shape}")

    logger.info("Step 3: Initializing model and loading weights...")
    model = HybridSN_BN(in_channels=cfg_dict['INPUT_CHANNELS'], out_channels=cfg_dict['NUM_CLASSES']).to(device)
    model_weights_path = os.path.join(cfg_dict['OUTPUT_PATH'], cfg_dict['MODEL_WEIGHTS_NAME'])
    if not os.path.exists(model_weights_path):
        logger.error(f"Model weights not found at {model_weights_path}. Please train the model first.")
        return
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.eval()
        logger.info(f"Model weights loaded successfully from {model_weights_path}")
    except Exception as e:
        logger.error(f"Error loading model weights from {model_weights_path}: {e}")
        return

    logger.info("Step 4: Preparing patches and collecting true labels for full map prediction...")
    patch_size = cfg_dict['PATCH_SIZE']
    margin = (patch_size - 1) // 2
    padded_X_processed = pad_with_zeros(X_processed, margin)
    labeled_pixels_row_col = np.argwhere(y_ground_truth > 0)
    predictions_for_labeled_pixels = []
    true_labels_for_labeled_pixels = []

    logger.info(f"Starting prediction for {len(labeled_pixels_row_col)} labeled pixels...")
    with torch.no_grad():
        for i in range(len(labeled_pixels_row_col)):
            r, c = labeled_pixels_row_col[i]
            patch = padded_X_processed[r : r + patch_size, c : c + patch_size, :]
            patch_tensor = torch.from_numpy(patch).float().to(device)
            patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            output = model(patch_tensor)
            _, pred_label = torch.max(output, 1)
            predictions_for_labeled_pixels.append(pred_label.item())
            true_labels_for_labeled_pixels.append(y_ground_truth[r, c] - 1)
            if (i + 1) % 2000 == 0:
                logger.info(f"Predicted {i+1}/{len(labeled_pixels_row_col)} pixels...")
    predictions_for_labeled_pixels = np.array(predictions_for_labeled_pixels)
    true_labels_for_labeled_pixels = np.array(true_labels_for_labeled_pixels)
    logger.info(f"Prediction finished for all {len(labeled_pixels_row_col)} labeled pixels.")

    logger.info("Step 5: Evaluating predictions for all labeled pixels...")
    if len(true_labels_for_labeled_pixels) > 0 and len(predictions_for_labeled_pixels) > 0:
        accuracy = accuracy_score(true_labels_for_labeled_pixels, predictions_for_labeled_pixels)
        unique_labels_in_data = np.unique(np.concatenate((true_labels_for_labeled_pixels, predictions_for_labeled_pixels))).astype(int)
        target_names = [f"Class {i}" for i in range(cfg_dict['NUM_CLASSES'])]

        report = classification_report(
            true_labels_for_labeled_pixels,
            predictions_for_labeled_pixels,
            digits=4,
            zero_division=0,
            labels=unique_labels_in_data,
            target_names=[target_names[i] for i in unique_labels_in_data if i < len(target_names)] # Ensure index is valid
        )
        cm = confusion_matrix(
            true_labels_for_labeled_pixels,
            predictions_for_labeled_pixels,
            labels=unique_labels_in_data
        )
        logger.info(f"Overall Accuracy on all labeled pixels: {accuracy:.4f}")
        logger.info(f"Classification Report on all labeled pixels:\n{report}")
        logger.info(f"Confusion Matrix on all labeled pixels (for observed labels):\n{cm}")
    else:
        logger.warning("No labeled pixels found or no predictions made to evaluate.")

    logger.info("Step 6: Plotting and saving classification map...")
    plot_classification_map(
        y_pred=predictions_for_labeled_pixels,
        y_gt=y_ground_truth,
        output_path=cfg_dict['OUTPUT_PATH'],
        dataset_name=cfg_dict['DATASET_NAME'],
        num_classes=cfg_dict['NUM_CLASSES'] # 传递类别数
    )
    logger.info("Prediction script finished.")

if __name__ == '__main__':
    predict_full_map(config)