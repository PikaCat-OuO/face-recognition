import cv2
import os

# 获取被摄影者的名字，创建文件夹
file_path = f'./faceImages/{input("请输入名字：")}/'
os.makedirs(file_path, exist_ok=True)

# 打开摄像头
camera = cv2.VideoCapture(700)
for i in range(600):
    # 利用摄像头抓取数据
    state, pic = camera.read()
    # 如果无法获取图片就抛出异常
    if not state:
        raise RuntimeError('无法从摄像头抓取图片')
    # 展示从摄像头抓取到的图片
    cv2.imshow('capture', pic)
    # 将图片保存到文件夹中
    cv2.imwrite(file_path + f'{i + 1}.jpg', pic)
    cv2.waitKey(1)

# 释放摄像头资源
camera.release()
# 结束所有的窗口
cv2.destroyAllWindows()
