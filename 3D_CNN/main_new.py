import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report
from Net import HybridSN_BN
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from processing_library import *


pca_components = 30
PCA = True
patch_size = 25
test_ratio = 0.95
random_state = 345
X = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_gt.mat')['indian_pines_gt']

# 保存包含背景值的y，以便画出整体预测图
y_full = y
X_pca = applyPCA(X, numComponents=pca_components, is_PCA=PCA)
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
Xtrain, Xtest, ytrain, ytest, data_test = splitTrainTestSet(X_pca, y, test_ratio, random_state, patch_size, pca_components,is_PCA=PCA)

trainset = TrainDS(Xtrain, ytrain)
testset = TestDS(Xtest, ytest)

# 创建 DataLoader
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False)


def train(net, is_train, model_path=None):
    current_loss_his = []
    current_Acc_his = []

    # 如果 model_path 没有传入，设置默认路径
    if model_path is None:
        model_path = 'E:/Project/Classification_Network/3D_CNN/best_model.pth'

    if is_train:
        best_net_wts = copy.deepcopy(net.state_dict())
        best_acc = 0.0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # 开始训练
        total_loss = 0
        for epoch in range(100):
            net.train()  # 将模型设置为训练模式
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 优化器梯度归零
                optimizer.zero_grad()
                # 正向传播 + 反向传播 + 优化 
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 每个epoch结束后，在验证集上测试一下当前模型的准确率
            net.eval()   # 将模型设置为验证模式
            current_acc = test_acc(net)
            current_Acc_his.append(current_acc)

            # 如果当前准确率大于最好的准确率，就更新最好的准确率和对应的网络参数
            if current_acc > best_acc:
                best_acc = current_acc
                best_net_wts = copy.deepcopy(net.state_dict())
                torch.save(best_net_wts, model_path)

            print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]  [current acc: %.4f]' 
                  % (epoch + 1, total_loss / (epoch + 1), loss.item(), current_acc))
            current_loss_his.append(loss.item())

        print('Finished Training')
        print("Best Acc:%.4f" % (best_acc))

        # load best model weights
        net.load_state_dict(best_net_wts)

    else:
        # 如果不进行训练，检查是否存在预训练权重文件
        if os.path.exists(model_path):
            print(f"Loading pre-trained weights from {model_path}")
            net.load_state_dict(torch.load(model_path))
            net.eval()   # 将模型设置为验证模式
            current_acc = test_acc(net)
            current_Acc_his.append(current_acc)
        else:
            raise FileNotFoundError(f"No pre-trained weights found at {model_path}. Please train the model first.")

    return net, current_loss_his, current_Acc_his
def test_acc(net):
  
  count = 0
  # 模型测试
  for inputs, _ in test_loader:
      inputs = inputs.to(device)
      outputs = net(inputs)
      outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
      if count == 0:
          y_pred_test =  outputs
          count = 1
      else:
          y_pred_test = np.concatenate( (y_pred_test, outputs) )

  # 生成分类报告
  classification = classification_report(ytest, y_pred_test, digits=4)
  print(classification)
  index_acc = classification.find('weighted avg')
  accuracy = classification[index_acc+17:index_acc+23]
  return float(accuracy)

# 测试GPU是否可用
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# 网络放到对应的设备上
net = HybridSN_BN().to(device)

# 训练
model_save_path = 'E:/Project/Classification_Network/3D_CNN/best_model.pth'

# 训练网络
net, current_loss_his, current_Acc_his = train(net, is_train=False, model_path=model_save_path)

