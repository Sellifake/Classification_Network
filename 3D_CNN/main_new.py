import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from processing_library import *


pca_components = 30
PCA = False
patch_size = 5
test_ratio = 0.98
random_state = 345
X = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('E:\\Project\\Classification_Network\\3D_CNN\\data\\Indian_pines_gt.mat')['indian_pines_gt']

X_pca = applyPCA(X, numComponents=pca_components, is_PCA=PCA)
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio,randomState=random_state)
pass
