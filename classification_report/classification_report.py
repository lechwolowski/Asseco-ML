from __future__ import print_function
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
import os
import pandas
from datetime import datetime
import io
import itertools
from packaging import version
from six.moves import range
import tensorflow as tf
import sklearn.metrics
from sklearn.metrics import classification_report

load_dir = os.getcwd()
model_name = 'best_cifar50.h5'

class_names = np.array([
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
])

class_names = class_names[1::2]

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
train_filter = y_train % 2 != 0
test_filter = y_test % 2 != 0
train_filter = train_filter.flatten()
test_filter = test_filter.flatten()
x_train = x_train[train_filter]
y_train = y_train[train_filter]
x_test = x_test[test_filter]
y_test = y_test[test_filter]
y_train = np.array(list(map(lambda x: (x - 1) / 2, y_train)))
y_test = np.array(list(map(lambda x: (x - 1) / 2, y_test)))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = keras.models.load_model(os.path.join(load_dir, model_name))

y_classes = model.predict(x_train).argmax(axis=-1)
true_classes = y_train.flatten().astype(int)
report = classification_report(true_classes, y_classes, target_names=class_names, output_dict=True)
df = pandas.DataFrame(report).transpose()
df.to_csv('classification_report/classification_report_train.csv')

y_classes = model.predict(x_test).argmax(axis=-1)
true_classes = y_test.flatten().astype(int)
report = classification_report(true_classes, y_classes, target_names=class_names, output_dict=True)
df = pandas.DataFrame(report).transpose()
df.to_csv('classification_report/classification_report_test.csv')