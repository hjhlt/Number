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

network = Net()

network_state_dict = torch.load('model.pth')
network.load_state_dict(network_state_dict)

network.eval()


def solve(image):
    # 使用PIL读取图片
    image = cv.imread(image)

    image = cv.resize(image, [28, 28])

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = image / 255

    image = [[1 - num for num in row] for row in image]

    image = np.array(image)

    image = torch.from_numpy(image).float()

    # 使用matplotlib显示图片
    # plt.imshow(image, cmap='gray')

    # 显示图片
    # plt.show()

    image = image.unsqueeze(0)

    image = image.unsqueeze(0)

    return image


for i in range(10):
    x = 'test'
    image_name = f"{i}.jpg"
    image = solve(image_name)
    output = network(image)
    pred = output.data.max(1, keepdim=True)
    print(pred)
