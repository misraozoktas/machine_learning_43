import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import kagglehub as kg
import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory

import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

#print("Path to dataset files:", path)

# Setting up file paths for training and testing
USER_PATH = path
train_dir = USER_PATH + r'/Training/'
test_dir = USER_PATH + r'/Testing/'

#labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

train_dataset = image_dataset_from_directory(
    train_dir,
    batch_size=64,
    label_mode='int'
)

test_dataset = image_dataset_from_directory(
    test_dir,
    batch_size=64,
    label_mode='int'
)

train_set = train_dataset.take(1)  #takes first batch


while len(train_set) < 300:
    x = train_dataset.take(1)
    train_set.append(x)


for images, labels in train_set:
    print("Images shape:", images.shape)  
    print("Labels shape:", labels.shape)

