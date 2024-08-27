import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="3D CNN Classification Network Parameters")

    # PCA components
    parser.add_argument('--pca_components', type=int, default=30,
                        help="Number of components to keep after applying PCA (default: 30)")

    # Toggle PCA usage
    parser.add_argument('--PCA', action='store_true', default=True,
                        help="Flag to determine whether to use PCA (default: True)")

    # Patch size
    parser.add_argument('--patch_size', type=int, default=25,
                        help="Size of the patch to be extracted from the input data (default: 25)")

    # Test ratio
    parser.add_argument('--test_ratio', type=float, default=0.90,
                        help="Ratio of data to be used for testing (default: 0.90)")

    # Random state for reproducibility
    parser.add_argument('--random_state', type=int, default=345,
                        help="Random state for reproducibility of results (default: 345)")

    # Training mode toggle
    parser.add_argument('--is_train', action='store_true', default=True,
                        help="Flag to determine whether to run in training mode (default: True)")

    # Plot results toggle
    parser.add_argument('--is_plot', action='store_true', default=True,
                        help="Flag to determine whether to plot results (default: True)")

    # Number of training epochs
    parser.add_argument('--train_epoch', type=int, default=100,
                        help="Number of epochs to train the model (default: 100)")

    # Path to data directory
    parser.add_argument('--data_path', type=str, default='E:\\Project\\Classification_Network\\3D_CNN\\data\\',
                        help="Path to the directory containing the dataset (default: 'E:\\Project\\Classification_Network\\3D_CNN\\data\\')")

    # Path to save or load weights
    parser.add_argument('--weight_path', type=str, default='E:/Project/Classification_Network/3D_CNN/',
                        help="Path to save or load model weights (default: 'E:/Project/Classification_Network/3D_CNN/')")

    args = parser.parse_args()
    return args

def train(net, is_train, model_path, train_eopch):
    current_loss_his = []
    current_Acc_his = []

    if is_train:
        best_net_wts = copy.deepcopy(net.state_dict())
        best_acc = 0.0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        total_loss = 0
        for epoch in range(train_eopch):
            net.train()  # 将模型设置为训练模式
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()  # 优化器梯度归零
                outputs = net(inputs)  # 正向传播
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 优化器步进
                total_loss += loss.item()

            net.eval()  # 将模型设置为验证模式
            current_acc = test_acc(net)
            current_Acc_his.append(current_acc)

            if current_acc > best_acc:  # 如果当前准确率更好，则保存模型权重
                best_acc = current_acc
                best_net_wts = copy.deepcopy(net.state_dict())
                torch.save(best_net_wts, model_path)

            print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]  [current acc: %.4f]' 
                  % (epoch + 1, total_loss / (epoch + 1), loss.item(), current_acc))
            current_loss_his.append(loss.item())

        print('Finished Training')
        print("Best Acc:%.4f" % (best_acc))

        net.load_state_dict(best_net_wts)  # 加载最佳模型权重

    else:
        if os.path.exists(model_path):
            print(f"Loading pre-trained weights from {model_path}")
            net.load_state_dict(torch.load(model_path))
            net.eval()  # 将模型设置为验证模式
            current_acc = test_acc(net)
            current_Acc_his.append(current_acc)
        else:
            raise FileNotFoundError(f"No pre-trained weights found at {model_path}. Please train the model first.")

    return net, current_loss_his, current_Acc_his

def test_acc(net):
    count = 0
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test =  outputs
            count = 1
        else:
            y_pred_test = np.concatenate( (y_pred_test, outputs) )

    classification = classification_report(ytest, y_pred_test, digits=4)
    index_acc = classification.find('weighted avg')
    accuracy = classification[index_acc+17:index_acc+23]
    return float(accuracy)

def full_data_test(net, is_plot):
    index = 0
    for inputs, _ in full_test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if index == 0:
            y_pred_full =  outputs
            index = 1
        else:
            y_pred_full = np.concatenate( (y_pred_full, outputs) )
    if is_plot:
        plot(y_pred_full=y_pred_full, y_full=y_full)
    return None

if __name__ == "__main__":
    args = parse_args()

    # Load data
    X = sio.loadmat(args.data_path + 'Indian_pines_corrected.mat')['indian_pines_corrected']
    y = sio.loadmat(args.data_path + 'Indian_pines_gt.mat')['indian_pines_gt']

    y_full = y  # 保存包含背景值的y，以便画出整体预测图
    X_pca = applyPCA(X, numComponents=args.pca_components, is_PCA=args.PCA)
    X_pca, y = createImageCubes(X_pca, y, windowSize=args.patch_size)
    Xtrain, Xtest, ytrain, ytest, data_test = splitTrainTestSet(X_pca, y, args.test_ratio, args.random_state, args.patch_size, args.pca_components, is_PCA=args.PCA)

    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    full_testset = FullTestDS(data_test, y)

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(dataset=full_testset, batch_size=128, shuffle=False)

    # 测试GPU是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # 初始化网络
    net = HybridSN_BN().to(device)

    # 训练
    model_save_path = args.weight_path + 'best_model.pth'
    net, current_loss_his, current_Acc_his = train(net, is_train=args.is_train, model_path=model_save_path, train_eopch=args.train_epoch)

    # 全数据集预测并画出预测图
    full_data_test(net, is_plot=args.is_plot)