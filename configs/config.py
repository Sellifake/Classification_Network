# configs/config.py

import torch

# -- 数据集和路径配置 --
# 数据集文件路径
DATA_PATH = './data/'
# 结果保存路径
OUTPUT_PATH = './outputs/'
# 数据集文件名
DATASET_NAME = 'Indian_pines'
# 模型权重保存的文件名
MODEL_WEIGHTS_NAME = 'final_model.pth' # 因为没有验证集了，可以改个名字

# -- 数据预处理配置 --
# 是否启用PCA降维
USE_PCA = True
# PCA降维后的主成分数量
PCA_COMPONENTS = 30
# 每个像素邻域提取的图像块（patch）大小，必须是奇数
PATCH_SIZE = 25

# -- 数据集划分配置 --
# 目标：10% 训练, 0% 验证, 90% 测试
# 1. 首先划分出90%的测试集
TEST_RATIO = 0.90
# 2. 剩余10%全部用于训练，不再划分验证集
VALIDATION_RATIO = 0.00 # 设置为0，表示不使用验证集
# 随机种子，用于保证实验结果的可复现性
RANDOM_STATE = 345

# -- 模型训练配置 --
# 训练的总轮次
EPOCHS = 100
# 批处理大小
BATCH_SIZE = 128
# 学习率
LEARNING_RATE = 0.001
# 分类类别数 (Indian Pines 数据集有16类，标签从0到15)
NUM_CLASSES = 16
# 输入通道数 (通常为1，因为3D卷积的输入是单通道的立方体)
INPUT_CHANNELS = 1

# -- 设备配置 --
# 自动选择可用设备 (GPU或CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")