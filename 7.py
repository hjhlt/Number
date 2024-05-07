import cv2
from PIL import Image, ImageTk
import tkinter
from test import solve
import numpy as np
import torch
from Net import Net

# 创建一个TK界面
root = tkinter.Tk()
root.geometry("1600x900")
root.resizable(False, False)
root.title('手写数字识别')

video = cv2.VideoCapture(0)
res = video.set(3, 1248)

network = Net()

network_state_dict = torch.load('model.pth')
network.load_state_dict(network_state_dict)


def pre(img):
    x, y, z = img.shape
    return img[x // 2 - 350:x // 2 + 349, y // 2 - 350:y // 2 + 349]


def fangda(img):
    result = np.zeros((700, 700))
    for i in range(28):
        for j in range(28):
            result[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = img[i][j]
    return result


cnt = 0


def imshow():
    global video
    global root
    global image1
    global image2
    global ans
    global cnt
    res, img = video.read()
    img = pre(img)

    if res == True:
        # 将adarray转化为image
        img1 = Image.fromarray(img)
        # 显示图片到label
        img1 = ImageTk.PhotoImage(img1)

        img2 = solve(img) * 255
        img2 = fangda(img2)
        img2 = Image.fromarray(img2)
        img2 = ImageTk.PhotoImage(img2)
        image1.image = img1
        image1['image'] = img1
        image2.image = img2
        image2['image'] = img2
        img3 = solve(img)
        image = torch.from_numpy(img3).float()
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        output = network(image)
        pred = output.data.max(1, keepdim=True)
        answer = pred[1][0][0].item()
        if pred[0][0][0].item() > -0.001:
            ans.config(text="数字为" + str(answer))
            cnt = 0
        else:
            cnt += 1
        if cnt > 60:
            ans.config(text="未识别到数字")

    # 创建一个定时器，每10ms进入一次函数
    root.after(50, imshow)


label1 = tkinter.Label(root, text='原图像')
label2 = tkinter.Label(root, text='预处理后的图像')

label1.place(x=0, y=0)
label2.place(x=900, y=0)
# 创建label标签
image1 = tkinter.Label(root, text=' ', width=700, height=700)
image1.place(x=0, y=25, width=700, height=700)

image2 = tkinter.Label(root, text=' ', width=700, height=700)
image2.place(x=900, y=25, width=700, height=700)

ans = tkinter.Label(root, text='未识别到数字')
ans.place(x=750, y=800, width=100, height=100)

imshow()

root.mainloop()

# 释放video资源
video.release()
