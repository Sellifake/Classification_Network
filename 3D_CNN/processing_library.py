import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def applyPCA(X, numComponents, is_PCA):

    print('\n... ... PCA tranformation ... ...')
    if is_PCA:
        print('进行PCA降维')
        # PCA函数要求输入数据为二维矩阵，因此需要将三维矩阵展平
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)

        # 将降维后的数据重新整形为三维矩阵
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        print('Data shape after PCA: ', newX.shape)
    else:
        # 如果不进行 PCA，则直接返回原始数据
        print('不进行PCA')
        newX = X
        print('Data shape after PCA: ', newX.shape)

    return newX

# 边缘像素补0
def padWithZeros(X, pad_margin):
    newX = np.zeros((X.shape[0] + 2 * pad_margin, X.shape[1] + 2* pad_margin, X.shape[2]))
    x_offset = pad_margin
    y_offset = pad_margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# 创建padding后的图像块
def createImageCubes(X, y, windowSize, removeZeroLabels = True):
    print('\n... ... create data cubes ... ...')
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, pad_margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    
    # 去掉标签为0的图像块，标签为0表示背景
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    print('Data cube X shape: ', patchesData.shape)
    print('Data cube y shape: ', patchesLabels.shape)
    return patchesData, patchesLabels

# 将数据集划分为训练集和测试集
def splitTrainTestSet(X, y, testRatio, randomState):
    print('\n... ... create train & test data ... ...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    print('Xtrain shape: ', X_train.shape)
    print('Xtest  shape: ', X_test.shape)
    return X_train, X_test, y_train, y_test