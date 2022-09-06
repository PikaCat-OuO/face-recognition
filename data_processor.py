import os
import random
import re
import cv2
import numpy as np
from shutil import copyfile
from net.mtcnn import mtcnn
from utils import utils
from tqdm import tqdm, trange


# 遍历文件夹下所有的图片，改名
def rename_images():
    base_url = 'faceImages'
    for folder_name in os.listdir(base_url):
        folder_url = base_url + f'/{folder_name}'
        for image_name in os.listdir(f'faceImages/{folder_name}'):
            image_url = folder_url + f'/{image_name}'
            file_extension = image_name[image_name.rfind('.'):]
            new_file_name = str(int(re.search(r'\d+', image_name).group(0))) + file_extension
            os.rename(image_url, folder_url + f'/{new_file_name}')


# 数据增强，随机选取数据进行复制操作
def data_enhancement():
    base_url = 'faceImages'
    for folder_name in os.listdir(base_url):
        folder_url = base_url + f'/{folder_name}'
        image_names = os.listdir(f'faceImages/{folder_name}')
        for i in range(len(image_names) + 1, 601):
            src_file_name = random.choice(image_names)
            file_extension = src_file_name[src_file_name.rfind('.'):]
            copyfile(folder_url + '/' + src_file_name, folder_url + f'/{i}{file_extension}')


# 识别图片中的人脸，将图片转换为灰度图
def image_gray_process():
    mtcnn_model = mtcnn()

    for cur_dir, _, file_names in os.walk('faceImages'):
        new_dir = cur_dir.replace('faceImages', 'faceImagesGray')
        os.makedirs(new_dir, exist_ok=True)
        for file_name in tqdm(file_names):
            img = cv2.imread(os.path.join(cur_dir, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 使用mtcnn模型检测，并将人脸对正并裁剪出来
            width, height, _ = np.shape(img)
            rectangles = utils.rect2square(mtcnn_model.detectFace(img, [0.5, 0.6, 0.8]).astype(np.int32))
            rectangles[0, [0, 2]] = np.clip(rectangles[0, [0, 2]], 0, width)
            rectangles[0, [1, 3]] = np.clip(rectangles[0, [1, 3]], 0, height)
            left, top, right, bottom = rectangles[0][:4]
            landmark = np.reshape(rectangles[0][5:15], (5, 2)) - np.array([left, top])
            crop_img = img[top: bottom, left: right]
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(new_dir, file_name), crop_img)


# 将图片数据转换成numpy数组
def image_to_numpy_array():
    training_data = []
    validation_data = []
    training_labels = []
    validation_labels = []

    # 处理数据，裁剪图片符合模型输入，标上标签
    for dir_name in os.listdir('faceImagesGray'):
        for i in trange(1, 601):
            crop_img = cv2.resize(cv2.imread(os.path.join('faceImagesGray', dir_name, f'{i}.jpg')), (160, 160))
            if i <= 480:
                training_data.append(crop_img)
                training_labels.append(dir_name)
            else:
                validation_data.append(crop_img)
                validation_labels.append(dir_name)

    # 保存为npy文件
    np.save('./data.npy', np.array(training_data + validation_data))
    np.save('./labels.npy', np.array(training_labels + validation_labels))


image_to_numpy_array()