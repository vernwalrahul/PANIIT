
# coding: utf-8

# In[37]:


import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
N_CLASS = 6


# In[115]:


ld = "training/simplified_training/"
X_train = []
for i in range(1, 5001):
    im = cv2.imread(ld+str(i)+".png", 0)
    X_train.append(im.reshape(1,200,200))
X_train = np.array(X_train)
Y_df = pd.read_csv("training/solution.csv")
Y_df = Y_df.values[:,1]
Y_train = np.zeros((len(X_train), N_CLASS))
Y_train[np.arange(len(X_train)), Y_df-1] = 1
X_train = X_train.astype('float32')
X_train /= 255

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[ ]:



batch_size = 128
num_classes = N_CLASS
epochs = 20
input_shape = X_train[0].shape
print("input_shape = ", input_shape)

model = keras.models.load_model("my_model.h5")

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
model.save("my_model"+str(epochs)+".h5")