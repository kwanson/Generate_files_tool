# 데이터 보기
import pandas as pd
import numpy as np
from glob import glob

# 이미지데이터 로딩
from PIL import Image
import cv2
from tqdm import tqdm

# 파일경로 설정
import os
import shutil
import json

# Modeling
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# GPU 환경 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Others
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

data_path = 'E:\\open'


train_path = data_path + '\\train'
test_path = data_path + '\\test'

print(train_path)

hand_gesture = pd.read_csv(data_path + '\\hand_gesture_pose.csv')
sample_submission = pd.read_csv(data_path + '\\sample_submission.csv')

# Train 데이터에 있는 폴더를 glob로 불러와
# sorted method를 통해 숫자 순으로 정렬합니다.
train_folders = sorted(glob(train_path + '/*'), key = lambda x : int(x.split('\\')[-1]))
test_folders  = sorted(glob(test_path + '/*'), key = lambda x : int(x.split('\\')[-1]))
#train_folders[:5]

train_folder = train_folders[2]

image_paths = sorted(glob(train_folder + '/*.png'), key = lambda x : int(x.split('\\')[-1].replace('.png','')))
json_path   = glob(train_folder + '/*.json')[0]

image_path = image_paths[0]
img = Image.open(image_path)
img_arr = np.array(img)
print(img_arr.shape)
plt.imshow(img_arr)
plt.axis('off')
plt.show()


js = json.load(open(json_path))

print("json keys              : ", js.keys())

print("json action info       : ",js.get('action'))

print("json actor info        : ",js.get('actor'))

print("json annotations keys  : ",js.get('annotations')[0].keys())

print(js.get('annotations')[1].get('data'))

answers = []
for train_folder in train_folders :
    json_path = glob(train_folder + '/*.json')[0]
    js = json.load(open(json_path))
    cat = js.get('action')[0]
    cat_name = js.get('action')[1]
    answers.append([train_folder.replace(data_path,''),cat, cat_name])

answers = pd.DataFrame(answers, columns = ['train_path','answer', 'answer_name'])

classes = pd.get_dummies(answers[['answer']], columns = ['answer']).to_numpy()


np.random.shuffle(train_folders) # 일반화 가능성 높이기

images  = []
targets = []
for train_folder in tqdm(train_folders) :
    image_paths = sorted(glob(train_folder + '/*.png'), key = lambda x : int(x.split('\\')[-1].replace('.png','')))
    query_path  = train_folder.replace(data_path,'')
    target = classes[int(train_folder.split('\\')[-1])]
    target
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        images.append(img)
        targets.append(target)


X = np.array(images)
print('Train X Shape : ', X.shape)

y = np.array(targets)
print('Train y Shape : ', y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      random_state=2021,
                                                      test_size=0.2,
                                                      stratify = y)

print('X_train shape : ', X_train.shape)
print('X_valid shape : ', X_valid.shape)
print('y_train shape : ', y_train.shape)
print('y_valid shape : ', y_valid.shape)


test_images  = []
for test_folder in tqdm(test_folders, total = len(test_folders)) :
    image_paths = sorted(glob(test_folder + '/*.png'), key = lambda x : int(x.split('\\')[-1].replace('.png','')))
    query_path  = test_folder.replace(data_path,'')
    test_image = []
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
    test_images.append(test_image)

test_images = np.array(test_images)
print(test_images.shape)


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
tf.test.is_gpu_available()


baseModel = VGG16(weights='imagenet', include_top=False)
baseModel.trainable = False

model_in = Input(shape = (224,224,3))
base_model = baseModel(model_in)
head_model = AveragePooling2D(pool_size=(7, 7))(base_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(512, activation="relu")(head_model)
head_model = Dropout(0.2)(head_model)
head_model = Dense(256, activation="relu")(head_model)
head_model = Dropout(0.2)(head_model)
model_out = Dense(classes.shape[1], activation="softmax")(head_model)

model = Model(inputs=model_in, outputs=model_out)

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data = (X_valid, y_valid),
          epochs=20,
          verbose = 1,
          batch_size=64)

predictions = []
for test_image in tqdm(test_images, total = len(test_images)) :
    prediction = np.mean(model.predict(np.array(test_image)), axis = 0)
    predictions.append(prediction)

sample_submission.iloc[:,1:] = predictions
#display(sample_submission.head())
sample_submission.to_csv('./BASELINE.csv', index=False)

#hello_1st