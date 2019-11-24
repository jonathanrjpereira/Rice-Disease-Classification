from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pathlib


train_data_dir = "C:/Users/Jonathan/Documents/GitHub/RiceDisease/Resize/train"
train_label_dir = pathlib.Path(train_data_dir)

test_data_dir = "C:/Users/Jonathan/Documents/GitHub/RiceDisease/Resize/test"
test_label_dir = pathlib.Path(test_data_dir)



CATEGORIES = np.array([item.name for item in train_label_dir.glob('*') if item.name != "LICENSE.txt"])
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
# print(train_labels)
# print(test_labels)
