import cv2
from PIL import Image, ImageTk
import tkinter

# 创建一个TK界面
root = tkinter.Tk()
root.geometry("640x480")
root.resizable(False, False)
root.title('手写数字识别')

video = cv2.VideoCapture(0)


def imshow():
    global video
    global root
    global image
    res, img = video.read()

    if res == True:
        # 将adarray转化为image
        img = Image.fromarray(img)
        # 显示图片到label
        img = ImageTk.PhotoImage(img)
        image.image = img
        image['image'] = img
    # 创建一个定时器，每10ms进入一次函数
    root.after(10, imshow)


# 创建label标签
image = tkinter.Label(root, text=' ', width=280, height=280)
image.place(x=0, y=0, width=280, height=280)

imshow()

root.mainloop()

# 释放video资源
video.release()
