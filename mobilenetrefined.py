# Use mobilenet_v2 classifier to classify random input images based on ImageNet Labels

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import datasets, layers, models

# keras = tf.keras
from tensorflow.keras import layers



print(tf.__version__)


import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pathlib
import random
from sklearn.utils import shuffle

# !pip install -U tf-hub-nightly
import tensorflow_hub as hub


train_data_dir = "C:/Users/Jonathan/Pictures/Scans/train"
train_label_dir = pathlib.Path(train_data_dir)



CATEGORIES = np.array([item.name for item in train_label_dir.glob('*') if item.name != "LICENSE.txt"])
class_names = CATEGORIES
print(CATEGORIES)



def createdataset(DATADIR, label_dir, CATEGORIES, img_size):
    image_count = len(list(label_dir.glob('*/*.jpg')))
    print(image_count)



    IMG_SIZE = img_size

    datalist = []

    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = np.where(CATEGORIES == category)

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                # img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                datalist.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
    return datalist


training_dataset = createdataset(train_data_dir, train_label_dir, CATEGORIES, 224)


print(len(training_dataset))



def dataset(datasets):
    xdata = []
    ylabels = []
    # random.shuffle(datasets)
    for datas,labels in datasets:
        xdata.append(datas)
        ylabels.append(labels)
    return xdata, ylabels

train_images, train_labels = dataset(training_dataset)

print(len(train_images))
print(len(train_labels))




train_images = np.array(train_images)

train_labels = np.array(train_labels)


print(train_images.shape)

train_images = train_images.reshape(train_images.shape[0], 224, 224, 3)


train_images = train_images / 255.0




classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))])
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
imageboi_labels = []


for i in range(25):
    imageboi = train_images[i]
    result = classifier.predict(imageboi[np.newaxis, ...])
    # print(result.shape)
    predicted_class = np.argmax(result[0], axis=-1)
    # print(predicted_class)
    predicted_class_name = imagenet_labels[predicted_class]
    imageboi_labels.append(predicted_class_name)

print(imageboi_labels)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(imageboi_labels[i])
plt.show()
