# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 21:35:21 2021

@author: Youstina
"""

#Importing Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import tensorflow as tf 

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten ,GaussianNoise
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

import numpy

import os
import cv2
import PIL
from PIL import Image
import seaborn
import glob
import pathlib


#Loading the data 
print(os.listdir("E:/ml/"))
benign_path="E:/ml/benign"
malignant_path="E:/ml/malignant"
benign_path = pathlib.Path(benign_path)
malignant_path = pathlib.Path(malignant_path)
benign = os.listdir(benign_path)
malignant = os.listdir(malignant_path)




train_data_images = {
    "benign" : list(benign_path.glob("*.png")),
    "malignant" : list(malignant_path.glob("*.png"))
}
train_labels = {
    "benign": 0, "malignant": 1
}
print(train_data_images["benign"])


X, y = [], []   # is the data and y is the labels
for label, images in train_data_images.items():
  for image in images:
    img = cv2.imread(str(image))   # Reading the image
    print(img)
    if img is not None:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (180, 180))
      X.append(img)
      y.append(train_labels[label])
print(X)



X_samp = numpy.array(X) 
y_samp = numpy.array(y)

np.save('Data', X_samp)
np.save('Labels', y_samp)

print('Cells : {} | labels : {}'.format(X_samp.shape , y_samp.shape))

X_samp = np.load("E:/ml/Data.npy")
y_samp = np.load("E:/ml/Labels.npy")


#ploting images and labels to understand the data
plt.figure(1 , figsize = (15 , 9)) 
n = 0 
for i in range(49):
    n += 1 
    r = np.random.randint(0 , X_samp.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(X_samp[r[0]])
    plt.title('{} : {}'.format('malignant' if y_samp[r[0]] == 1 else 'benign' ,
                               y_samp[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()


# normalization
X_samp = (X_samp/255) 
X_samp[0].shape

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_samp, y_samp, test_size=0.2)
print(X_train.shape)


"""# Data Argumentation"""

data_argumentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomContrast(0.3),
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
    keras.layers.experimental.preprocessing.RandomZoom(0.5) 
])

# for i in range(5):
#     plt.imshow(data_argumentation(X_train[0]))
#     plt.show()

"""# Model Building"""

model = Sequential()
model.add(Flatten(input_shape=(180, 180, 3)))
model.add(Dense(260,activation='linear'))
model.add(Dense(290,activation='relu'))
model.add(Dense(290,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# # configure a model for mean-squared error regression
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(filepath="weights.h5", verbose=1, save_best_only=True)


# # # Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2, callbacks=[checkpoint])
model.summary()
model.evaluate(X_test, y_test)



print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()







