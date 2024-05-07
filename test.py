import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from Net import Net
import numpy as np
import tkinter

network = Net()

network_state_dict = torch.load('model.pth')
network.load_state_dict(network_state_dict)

network.eval()


def maxPool(img):
    img = np.array(img)
    x, y = img.shape
    result = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            result[i, j] = np.min(img[i * x // 28:(i + 1) * x // 28, j * y // 28:(j + 1) * y // 28])
            if result[i, j] > 0.3:
                result[i, j] = 1
            else:
                result[i, j] = 0
    return result


def solve(image):
    # 使用PIL读取图片
    # image = cv.imread(image)
    # 将图片变为灰度图片
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 将像素值变为0-1之间方便神经网络进行预测
    image = image / 255
    # 进行最大池化特征提取并将图片压缩到28x28的大小
    image = maxPool(image)
    # 将黑白颠倒
    image = [[1 - num for num in row] for row in image]
    image = np.array(image)
    # image = torch.from_numpy(image).float()
    # 使用matplotlib显示图片
    # plt.imshow(image, cmap='gray')
    # 显示图片
    # plt.show()
    # image = image.unsqueeze(0)
    # image = image.unsqueeze(0)
    return image

# for i in range(1):
#     x = 'test'
#     image_name = "shouxie.jpg"
#     image = solve(image_name)
#     output = network(image)
#     pred = output.data.max(1, keepdim=True)
#     print(pred)
