import cv2
import os
import numpy as np
from utils import utils
from net.mtcnn import mtcnn
from net.inception import InceptionResNetV1

# 加载mtcnn模型
mtcnn_model = mtcnn()
# 加载facenet模型
facenet = InceptionResNetV1()
facenet.load_weights('./model/facenet.h5')

# 准备好所有人的人脸
all_faces = sorted(os.listdir('faceImages'))

# 打开摄像头
camera = cv2.VideoCapture(700)
while True:
    # 利用摄像头抓取数据
    state, pic = camera.read()
    # 如果无法获取图片就抛出异常
    if not state:
        raise RuntimeError('无法从摄像头抓取图片')

    # 转换成RGB图片
    img = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    # 使用mtcnn模型检测，并将人脸对正并裁剪出来
    width, height, _ = np.shape(img)
    rectangles = mtcnn_model.detectFace(img, [0.5, 0.6, 0.8]).astype(np.int32)
    if len(rectangles):
        rectangles = utils.rect2square(rectangles)
        rectangles[0, [0, 2]] = np.clip(rectangles[0, [0, 2]], 0, width)
        rectangles[0, [1, 3]] = np.clip(rectangles[0, [1, 3]], 0, height)
        left, top, right, bottom = rectangles[0][:4]
        landmark = np.reshape(rectangles[0][5:15], (5, 2)) - np.array([left, top])
        crop_img = img[top: bottom, left: right]
        crop_img, _ = utils.Alignment_1(crop_img, landmark)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
        crop_img = utils.pre_process(np.expand_dims(cv2.resize(crop_img, (160, 160)), 0))
        # 放入facenet进行识别
        result = all_faces[np.argmax(facenet.predict(crop_img, batch_size=1))]
        cv2.rectangle(pic, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pic, result, (left, bottom - 15), font, 0.75, (255, 255, 255), 2)

    # 展示从摄像头抓取到的图片
    cv2.imshow('capture', pic)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 释放摄像头资源
camera.release()
# 结束所有的窗口
cv2.destroyAllWindows()
