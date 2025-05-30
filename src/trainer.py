# src/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader

from src.utils import setup_logger

class Trainer:
    """
    封装了模型的训练、评估和保存等所有相关逻辑。
    此版本不使用验证集，在训练结束后保存最终模型。
    """
    def __init__(self,
                 model: nn.Module,
                 config_dict: Dict[str, Any],
                 device: torch.device,
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model.to(device)
        self.config = config_dict
        self.device = device
        self.logger = setup_logger(
            log_dir=self.config['OUTPUT_PATH'],
            log_filename=f"{self.config['DATASET_NAME']}_trainer_log.txt" # 修改日志名
        )

        if class_weights is not None and torch.is_tensor(class_weights) and class_weights.numel() > 0 :
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            self.logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights.tolist()}")
        else:
            if class_weights is not None:
                 self.logger.warning("Class weights were provided but invalid. Using standard CrossEntropyLoss.")
            self.criterion = nn.CrossEntropyLoss()
            self.logger.info("Using standard CrossEntropyLoss.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['LEARNING_RATE'])
        self.model_save_path = os.path.join(self.config['OUTPUT_PATH'], self.config['MODEL_WEIGHTS_NAME'])
        os.makedirs(self.config['OUTPUT_PATH'], exist_ok=True)

    def train(self, train_loader: DataLoader): # 不再需要 val_loader
        """
        执行完整的训练流程。在训练结束后保存模型。
        """
        self.logger.info("----------- Starting Training (No Validation) -----------")

        for epoch in range(self.config['EPOCHS']):
            self.model.train()
            total_loss = 0.0
            correct_train_samples = 0
            total_train_samples = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train_samples += torch.sum(preds == labels.data)
                total_train_samples += inputs.size(0)

            avg_train_loss = total_loss / total_train_samples if total_train_samples > 0 else 0
            train_acc = correct_train_samples.double() / total_train_samples if total_train_samples > 0 else 0

            self.logger.info(
                f"Epoch [{epoch+1}/{self.config['EPOCHS']}] | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}"
            )

        # 训练结束后保存最终模型
        torch.save(self.model.state_dict(), self.model_save_path)
        self.logger.info(f"----------- Finished Training -----------")
        self.logger.info(f"Final model saved to {self.model_save_path}")

    def evaluate(self, data_loader: DataLoader, log_report: bool = True) -> Tuple[float, str]:
        """
        在给定的数据集上评估模型性能 (用于测试集)。
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if not all_labels:
            self.logger.warning("Evaluate method called with empty data_loader or no labels found.")
            return 0.0, "No samples to evaluate."

        accuracy = accuracy_score(all_labels, all_preds)
        unique_labels_in_data = np.unique(np.concatenate((all_labels, all_preds))).astype(int)
        target_names = [f"Class {i}" for i in range(self.config.get('NUM_CLASSES', 16))] # 获取类别名

        report = classification_report(
            all_labels, 
            all_preds, 
            digits=4, 
            zero_division=0, 
            labels=unique_labels_in_data, # 只报告实际出现的标签
            target_names=[target_names[i] for i in unique_labels_in_data] # 对应的类别名
            )

        if log_report:
            cm = confusion_matrix(all_labels, all_preds, labels=unique_labels_in_data)
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Classification Report:\n{report}")
            self.logger.info(f"Confusion Matrix (for observed labels):\n{cm}")
        return accuracy, report