# 高光谱图像3D2D混合分类网络
### 1. 数据读取
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
    - 最终变量 `X` 的大小为 (145, 145, 200)，最大值9604，最小值905，表示了IP数据集的实际值；`y` 的大小为 (145, 145)，最大值16，最小值0，表示了每一个像素的真实类别标签，其中标签0为背景标签，1-16为有效类别。
    - 统计变量中的不同值，以数组的形式返回:
        ```python
        unique_labels = np.unique(y)  # y的返回值应该是array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        ```

---

### 2. 对数据进行预处理
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
    - PCA，即主成分分析，常用数据降维方法。
    - PCA只改变了数据的波段数，不改变数据类型，即 `X` 的大小变为 (145, 145, numComponents)。


