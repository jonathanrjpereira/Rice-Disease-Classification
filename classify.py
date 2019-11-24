from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

print(tf.__version__)


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pathlib
import random


train_data_dir = "C:/Users/Jonathan/Documents/GitHub/RiceDisease/Resize/train"
train_label_dir = pathlib.Path(train_data_dir)

test_data_dir = "C:/Users/Jonathan/Documents/GitHub/RiceDisease/Resize/test"
test_label_dir = pathlib.Path(test_data_dir)



CATEGORIES = np.array([item.name for item in train_label_dir.glob('*') if item.name != "LICENSE.txt"])
class_names = CATEGORIES
print(CATEGORIES)



def createdataset(DATADIR, label_dir, CATEGORIES, img_size):
    image_count = len(list(label_dir.glob('*/*.jpg')))
    print(image_count)

    # for category in CATEGORIES:  # do dogs and cats
    #     path = os.path.join(DATADIR,category)  # create path to dogs and cats
    #     for img in os.listdir(path):  # iterate over each image per dogs and cats
    #         img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
    #     #     break  # we just want one for now so break
    #     # break  #...and one more!

    IMG_SIZE = img_size

    datalist = []

    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = np.where(CATEGORIES == category)

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                datalist.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
    return datalist


training_dataset = createdataset(train_data_dir, train_label_dir, CATEGORIES, 200)
testing_dataset = createdataset(test_data_dir, test_label_dir, CATEGORIES, 200)

print(len(training_dataset))
print(len(testing_dataset))


def dataset(datasets):
    xdata = []
    ylabels = []
    # random.shuffle(datasets)
    for datas,labels in datasets:
        xdata.append(datas)
        ylabels.append(labels)
    return xdata, ylabels

train_images, train_labels = dataset(training_dataset)
test_images, test_labels = dataset(testing_dataset)
print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))



train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(train_images.shape)
print(test_images.shape)

train_images = train_images.reshape(train_images.shape[0], 200, 200, 1)
test_images = test_images.reshape(test_images.shape[0], 200, 200, 1)

# print(train_labels)
# print(test_labels)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(200, 200)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(4, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=10)
#
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print('\nTest accuracy:', test_acc)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
