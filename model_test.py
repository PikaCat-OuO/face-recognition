import os
import numpy as np
from utils import utils
from tqdm import trange
from net.inception import InceptionResNetV1

# 加载facenet模型
facenet = InceptionResNetV1()
facenet.load_weights('./model/facenet.h5')

# 准备好所有人的人脸
all_faces = sorted(os.listdir('faceImages'))

acc = 0
total = 0

validation_data = np.load('data.npy')[4800:]
validation_labels = np.load('labels.npy')[4800:]

for i in trange(1200):
    crop_img = np.expand_dims(utils.pre_process(validation_data[i]), 0)
    # 放入facenet进行识别
    result = all_faces[np.argmax(facenet.predict(crop_img, batch_size=1))]
    acc += int(result == validation_labels[i])
    total += 1
    print(f'accuracy:{acc / total:.2%}')
