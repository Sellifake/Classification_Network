# src/utils.py

import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Optional 


INDIAN_PINES_COLORS = [
    '#0000FF',  # 0 (Blue) - 背景
    '#FF0000',  # 1 (Red)
    '#006400',  # 2 (DarkGreen)
    '#800080',  # 3 (Purple)
    '#FFA500',  # 4 (OrangeYellowish, from image it's more like orange)
    '#FFC0CB',  # 5 (Pink)
    '#98FB98',  # 6 (PaleGreen)
    '#A52A2A',  # 7 (Maroon/Brown)
    '#808080',  # 8 (Gray)
    '#00FFFF',  # 9 (Cyan/Aqua)
    '#FF00FF',  # 10 (Magenta/Fuchsia)
    '#00FF00',  # 11 (Lime/BrightGreen)
    '#D2691E',  # 12 (Chocolate/LighterBrown)
    '#ADD8E6',  # 13 (LightBlue)
    '#B0C4DE',  # 14 (LightSteelBlue/PalerBlue)
    '#FF8C00',  # 15 (DarkOrange)
    '#FFD700'   # 16 (Gold/DarkYellow)
]


def setup_logger(log_dir: str = "outputs", log_filename: str = "training_log.txt") -> logging.Logger:
    """
    配置并返回一个日志记录器(logger)，同时输出到控制台和文件。
    """
    logger = logging.getLogger('HybridSN_Logger')
    logger.setLevel(logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def plot_classification_map(
        y_pred: np.ndarray,
        y_gt: np.ndarray,
        output_path: str,
        dataset_name: str,
        num_classes: int = 16 # Indian Pines specific, 0 is background
    ) -> None:
    """
    根据预测结果和真实标签(GT)绘制并保存分类结果图，使用指定的颜色方案。

    Args:
        y_pred (np.ndarray): 模型对带标签像素的预测结果 (一维数组, 类别从0到N-1)。
        y_gt (np.ndarray): 原始的地面真实标签图 (二维 HxW 数组)，包含背景(0)。
        output_path (str): 结果图保存的目录。
        dataset_name (str): 数据集名称，用于文件名。
        num_classes (int): 数据集中的实际地物类别数量 (不包括背景)。
    """
    logger = logging.getLogger('HybridSN_Logger')
    prediction_map_display = np.zeros_like(y_gt, dtype=int) # 用于显示的图，背景为0
    
    labeled_pixels_coords = np.argwhere(y_gt > 0) # GT中类别从1到num_classes
    
    for i, (r, c) in enumerate(labeled_pixels_coords):
        # y_pred 中的标签是 0 到 num_classes-1
        # 在显示时，我们需要将这些标签映射回 GT 的标签体系（1到num_classes）
        # 或者，我们直接使用0到num_classes的颜色方案，其中0代表背景
        prediction_map_display[r, c] = y_pred[i] + 1 # 将0-15的预测映射到1-16以匹配颜色条（如果颜色条0是背景）

    # 使用截图中的颜色方案
    # 背景是蓝色 (对应标签0)，其他类别从1开始
    # 我们有17个颜色，对应标签0到16
    cmap = mcolors.ListedColormap(INDIAN_PINES_COLORS)
    
    # 创建一个从0到num_classes的归一化器 (共 num_classes + 1 个颜色)
    # 对应颜色条从0到16
    bounds = np.arange(num_classes + 2) - 0.5 # e.g., for 16 classes -> -0.5 to 16.5 for 17 colors
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8, 6)) # 调整图像大小以适应截图比例
    plt.imshow(prediction_map_display, cmap=cmap, norm=norm)
    
    # 设置颜色条，使其与截图一致
    # 颜色条的刻度应为0到16
    cbar = plt.colorbar(ticks=np.arange(num_classes + 1), spacing='proportional', orientation='vertical')
    cbar.set_label('Class', rotation=270, labelpad=15)
    
    plt.title("Classification Result", fontsize=14)
    plt.xlabel("Pixel Column", fontsize=12) # 示例x轴标签
    plt.ylabel("Pixel Row", fontsize=12)    # 示例y轴标签
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 移除坐标轴刻度值，但保留标签，以更接近截图风格 (可选)
    # plt.xticks([])
    # plt.yticks([])
    # 或者只显示主要刻度
    
    save_path = os.path.join(output_path, f"{dataset_name}_classification_map_custom_color.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Classification map with custom colors saved to {save_path}")
    plt.show()