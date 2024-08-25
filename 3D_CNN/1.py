import scipy.io as sio
import os
import numpy as np
X = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_gt.mat')['indian_pines_gt']

# print X，y中的最大值和最小值
print(np.max(X), np.min(X))
print(np.max(y), np.min(y))
# 统计y中不同值的个数
print(np.unique(y))
pass