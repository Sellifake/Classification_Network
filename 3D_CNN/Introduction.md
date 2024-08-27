# 高光谱图像3D2D混合分类网络
### 只记载了我在看代码时不懂的地方，以及一些个人理解
## 1. 数据读取
- **读取高光谱图像数据**:
    ```python
    X = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_corrected.mat')['indian_pines_corrected']
    ```
- **读取标签数据**:
    ```python
    y = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_gt.mat')['indian_pines_gt']
    ```
- **说明**:
    - 采用绝对路径读取，在VSCode中，运行路径默认是打开文件夹的路径，因此采用绝对路径更稳妥。
    - 原始数据（`.mat`）文件中包含数据主题、名称、版本等信息，需要具体到数组（`array`）里面。
    - 最终变量 `X` 的大小为 `(145, 145, 200)`，最大值`9604`，最小值`905`，表示了IP数据集的实际值；`y` 的大小为 `(145, 145)`，最大值`16`，最小值`0`，表示了每一个像素的真实类别标签，其中标签`0`为背景标签，`1-16`为有效类别。
    - 统计变量中的不同值，以数组的形式返回:
        ```python
        unique_labels = np.unique(y)  # y的返回值应该是`array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])`
        ```

---

## 2. 对数据进行预处理
- **应用PCA进行数据预处理**:
    ```python
    def applyPCA(X, numComponents, is_PCA=True):
        if is_PCA:
            newX = np.reshape(X, (-1, X.shape[2]))
            pca = PCA(n_components=numComponents, whiten=True)
            newX = pca.fit_transform(newX)
            newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
            print('Data shape after PCA: ', newX.shape)
        else:
            newX = X
            print('Data shape after PCA: ', newX.shape)
        return newX
    ```
- **说明**:
    - `PCA`，即主成分分析，常用数据降维方法。
    - `PCA`只改变了数据的波段数，不改变数据类型，即 `X` 的大小变为 `(145, 145, numComponents)`。

---

## 3. 对数据进行填充
- **在每个像素周围提取 `patch`，然后创建符合 `keras` 处理的格式**:
    ```python
    def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
        # 给 X 做 padding
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = padWithZeros(X, margin=margin)
        
        # 分割 patches
        patchesData = np.zeros(((X.shape[0] * X.shape[1]), windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchIndex = 0
        
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1

        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1
        
        return patchesData, patchesLabels
    ```

- **说明**:
    - 填充目的：避免卷积之后尺寸缩小，充分利用边缘像素的点。
    - 填充方法：通过在图像的边缘添加0填充的像素行和列，来扩展图像的大小。零填充最为常见。
    - 在创建Cube的过程中，移除标签值为0的背景样本，最终 `patchesData` 的大小是 `(10249, 5, 5, 30)`。

---

## 4. 将数据划分为训练集和测试集
- **划分数据集**:
    ```python
    def splitTrainTestSet(X, y, testRatio, randomState):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
        return X_train, X_test, y_train, y_test
    ```
- **说明**:
    - `stratify` 参数确保训练集和测试集各类别的比例与原始数据集中的比例一致。例如选择`5%`的样本作为训练样本，那么在训练集中每一类都会有`5%`的样本被选为训练，确保了数据的类别平衡。

---

## 5. 调整数据形状以适应Keras和PyTorch要求

- **调整 `X_train` 和 `X_test` 的形状**:
    ```python
    # 改变 X_train, Y_train 的形状，以符合 keras 的要求
    X_train = X_train.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_test  = X_test.reshape(-1, patch_size, patch_size, pca_components, 1)
    
    # 为了适应 pytorch 结构，数据要做 transpose
    X_train = X_train.transpose(0, 4, 3, 1, 2)
    X_test  = X_test.transpose(0, 4, 3, 1, 2)
    ```

- **说明**:
    - 调整后的 `X_train` 的大小为 `(1024, 1, 30, 25, 25)`。
    - 首先通过 `reshape` 将数据调整为 Keras 要求的形状，然后通过 `transpose` 进一步转换为适应 PyTorch 的格式。

---

## 6. 创建训练和测试数据集的加载器

- **训练数据集和测试数据集的定义**:
    ```python
    """ Training dataset"""
    class TrainDS(torch.utils.data.Dataset): 
        def __init__(self):
            self.len = X_train.shape[0]
            self.x_data = torch.FloatTensor(X_train)
            self.y_data = torch.LongTensor(y_train)        
        def __getitem__(self, index):
            # 根据索引返回数据和对应的标签
            return self.x_data[index], self.y_data[index]
        def __len__(self): 
            # 返回数据集的大小
            return self.len

    """ Testing dataset"""
    class TestDS(torch.utils.data.Dataset): 

    # 创建 train_loader 和 test_loader
    trainset = TrainDS()
    testset  = TestDS()
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=128, shuffle=False)
    ```

- **说明**:
    - 这段代码定义了两个数据集类 `TrainDS` 和 `TestDS`，分别用于训练数据集和测试数据集。
    - 主要功能是将已经处理过的 `X_train` 和 `y_train` 数据、`X_test` 和 `y_test` 数据转换为 PyTorch 可以使用的 `Dataset` 对象，以便在模型训练和测试中进行批量处理。
    
    - `__getitem__` 是一个特殊方法，它允许通过索引来访问数据集中的元素。在这里，它返回给定索引的输入数据和对应的标签，供模型使用。
    
    - `torch.utils.data.DataLoader` 是 PyTorch 中的一个类，它包装了一个 `Dataset` 对象，提供了对数据的批量加载、打乱顺序、并行加载等功能。通过设置 `batch_size`，可以控制每次从数据集中提取多少样本，`shuffle=True` 表示每个 epoch 结束后会打乱数据集的顺序，从而提高模型的泛化能力。
    - `Python`调用函数时，位置参数必须出现在关键字参数之前。
    - `batch_size`意味着模型每次梯度更新（即每次 `optimizer.step()`）时，会使用 128 个样本计算损失并更新模型的参数

---

## 7. 模型训练

- **模型训练函数 `train` 的定义**:
    ```python
    def train(net):

    ```

- **说明**:
    - `.train()` 和 `.eval()` 是 `nn.Module` 的内置方法，不需要单独定义。`train()` 用于将模型设置为训练模式，启用训练所需的操作，如 dropout 和 batch normalization；`eval()` 用于将模型设置为验证或测试模式，禁用这些操作，以保证在验证或测试时的一致性。
    - 在每个 epoch 的训练结束后，模型会在验证集上进行测试，并根据测试结果的准确率决定是否更新保存的模型权重。程序会保存在验证集上取得最高精度时的模型权重，以保证最终的模型是最优的。

---


## 7. 网络的改进

- **在原有网络基础上加入了BN（Batch Normalization）**:
    ```python
    self.conv3d_features = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=(7, 3, 3)),
        nn.BatchNorm3d(8),
        nn.ReLU(),
        nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
        nn.BatchNorm3d(16),
        nn.ReLU(),
        nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
        nn.BatchNorm3d(32),
        nn.ReLU()
    )
    ```

- **说明**:
    - 在这个改进的网络中，加入了 `Batch Normalization`（BN）层。`BatchNorm3d` 用于在3D卷积神经网络中进行批归一化。
    - `BN` 的主要作用是在每一层的输出上应用归一化，从而减轻梯度消失或爆炸的现象，并使得网络训练更加稳定、收敛速度更快。此外，BN还有助于减少对初始权重的敏感性，提升模型的泛化能力。
