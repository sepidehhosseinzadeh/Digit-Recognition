import os
import numpy as np
from tempfile import TemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import model_selection
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
import csv
import urllib2
import urllib, cStringIO
from PIL import Image

path = '/home/sepideh/MultiSpectralImages/'
name_file = 'ba295b7e-7893-4ae3-a2f7-54c5f0223e45_uofa'
data_file = pd.read_csv(path+name_file+'.csv')

wrong_files= []
X_tmp = []

input_size = 28
input_channels = 1
num_chanel = 10

for i in range(len(data_file.iloc[:,0])):
    try:
        url = data_file.iloc[i,1]
        response = urllib2.urlopen(url)
        data = response.read()
        X_i = csv.DictReader(data)
        for chanel in range(num_chanel):
            file = cStringIO.StringIO(urllib.urlopen(data_file.iloc[i,2+chanel]).read())
            img = np.asarray(Image.open(file))
            img_resize = cv2.resize(img, (input_size, input_size), interpolation = cv2.INTER_AREA)
            x = np.array(img_resize)
            X_tmp.append(x)
    except IOError:
        wrong_files += [i]


number_of_img = len(data_file.iloc[:,0])-len(wrong_files)
X_nparray = np.array(X_tmp) / 255
X = X_nparray.reshape((number_of_img, num_chanel, input_size , input_size))
labels_t = data_file[data_file.loc[:,' CSVFile'] != str(data_file.iloc[wrong_files,1].values)[2:-2]]
class_lable = LabelEncoder()
y = class_lable.fit_transform(labels_t.iloc[:,0].values)

encoder = OneHotEncoder()
Y = encoder.fit_transform(y.reshape(-1,1)).toarray()

X_out = TemporaryFile()
Y_out = TemporaryFile()
np.save(X_out, X)
np.save(Y_out, Y)

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(X, Y,
                                                                                   test_size = 0.2, random_state = 0)
batch_size = 120
num_classes = 20
epochs = 400
input_shape = (num_chanel, input_size, input_size)
l1_lambda = 0.00003


model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

opt = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
checkpoint = ModelCheckpoint(filepath=os.path.join(path, "model-{epoch:02d}.h5"))

model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_labels),
          callbacks=[tensorboard, checkpoint])


score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


