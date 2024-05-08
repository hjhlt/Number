import cv2
from PIL import Image, ImageTk
import tkinter
from pre import solve
import numpy as np
import torch
from Net import Net
from tkinter import filedialog
from tkinter import font

# 创建一个TK界面
root = tkinter.Tk()
root.geometry("1600x900")
root.resizable(False, False)
root.title('手写数字识别')

video = cv2.VideoCapture(0)
res = video.set(3, 1280)

network = Net()

network_state_dict = torch.load('./model/model.pth')
network.load_state_dict(network_state_dict)
network.eval()

def pre(img):
    x, y, z = img.shape
    return img[x // 2 - 350:x // 2 + 349, y // 2 - 350:y // 2 + 349]


def enlarge(img):
    result = np.zeros((700, 700))
    for i in range(28):
        for j in range(28):
            result[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = img[i][j]
    return result


cnt = 0
mode = 0
first = 1

def imshow():
    global video
    global root
    global image1
    global image2
    global ans
    global cnt
    global mode
    r, img = video.read()
    img = pre(img)

    if r and mode == 1:
        # 将adarray转化为image
        img1 = Image.fromarray(img)
        # 显示图片到label
        img1 = ImageTk.PhotoImage(img1)

        img2 = solve(img) * 255
        img2 = enlarge(img2)
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


def open_image():
    global video
    global root
    global image1
    global image2
    global ans
    global cnt
    global mode
    # 使用filedialog打开文件选择对话框
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        mode = 0
        # 使用Pillow读取图片
        img = Image.open(file_path)
        # 使用BICUBIC作为重采样过滤器
        img = img.resize((700, 700), Image.BICUBIC)
        # 将PIL Image对象转换为Tkinter可以显示的PhotoImage对象
        photo = ImageTk.PhotoImage(img)

        img = np.array(img)
        img2 = solve(img) * 255
        img2 = enlarge(img2)
        img2 = Image.fromarray(img2)
        img2 = ImageTk.PhotoImage(img2)

        img3 = solve(img)
        image = torch.from_numpy(img3).float()
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        output = network(image)
        pred = output.data.max(1, keepdim=True)
        answer = pred[1][0][0].item()

        # 将上传的图片显示到窗口中
        image1.image = photo
        image1['image'] = photo
        image2.image = img2
        image2['image'] = img2

        if answer > -0.001:
            ans.config(text="数字为" + str(answer))
        else:
            ans.config(text="未识别到数字")


def start():
    global mode
    global first
    mode = 1
    if first == 1:
        imshow()
        first = 0

label1 = tkinter.Label(root, text='原图像')
label2 = tkinter.Label(root, text='预处理后的图像')

label1.place(x=0, y=0)
label2.place(x=900, y=0)
# 创建label标签
image1 = tkinter.Label(root, text=' ', width=700, height=700)
image1.place(x=0, y=25, width=700, height=700)

image2 = tkinter.Label(root, text=' ', width=700, height=700)
image2.place(x=900, y=25, width=700, height=700)

my_font = font.Font(size=15)
ans = tkinter.Label(root, text='未识别到数字',font=my_font)
ans.place(x=750, y=775, width=200, height=100)

btn_open = tkinter.Button(root, text="上传图片", command=open_image)
btn_open.place(x=200, y=800, width=100, height=50)
btn_video = tkinter.Button(root, text="使用摄像头捕捉", command=start)
btn_video.place(x=0, y=800, width=100, height=50)


root.mainloop()

# 释放video资源
video.release()
