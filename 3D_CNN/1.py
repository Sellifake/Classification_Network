import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
X = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_gt.mat')['indian_pines_gt']

your_array = y  # 请将此处替换为你的 145x145 uint8 数组

cmap = mcolors.ListedColormap([
    'blue',      # 类别0，蓝色
    'red',       # 类别1
    'green',     # 类别2
    'yellow',    # 类别3
    'purple',    # 类别4
    'orange',    # 类别5
    'pink',      # 类别6
    'brown',     # 类别7
    'gray',      # 类别8
    'cyan',      # 类别9
    'magenta',   # 类别10
    'lime',      # 类别11
    'darkred',   # 类别12
    'lightblue', # 类别13
    'lightgreen',# 类别14
    'darkorange',# 类别15
    'gold'       # 类别16
])

# 定义边界，使用类别编号作为边界
bounds = np.arange(18) - 0.5  # 17个类别，需要18个边界值
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# 绘制 your_array 图像
plt.figure(figsize=(8, 8))
plt.imshow(your_array, cmap=cmap, norm=norm)
plt.colorbar(ticks=np.arange(17))  # 添加颜色条，显示类别编号
plt.title("Segmentation Result")
plt.show()