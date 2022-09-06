import gc
import os
import numpy as np
from utils import utils
from net.inception import InceptionResNetV1
from keras.utils import to_categorical

# 读入准备好的训练数据，进行预处理
all_labels = sorted(os.listdir('faceImages'))

data = np.array(list(map(lambda x: utils.pre_process(x), np.load('data.npy'))))
labels = to_categorical(list(map(lambda x: all_labels.index(x), np.load('labels.npy'))))

# 建立模型
model = InceptionResNetV1()
model.compile(loss='categorical_crossentropy', optimizer='adagrad')
model.summary()

# 展开训练
model.fit(data, labels, epochs=100, batch_size=60, validation_split=0.2)

# 保存权重
model.save_weights('./model/facenet.h5')
