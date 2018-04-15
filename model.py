'''
model.py containing the script to create and train the model
'''

import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Cropping2D

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, AveragePooling2D



'''
Load training images and their labels/steering measurements
'''
# read labels from csv
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# use opencv to read images
images = []
measurements = []
for line in lines[1:]: # delete the first line which is the name of the column
    for i in range(3): # use Multiple Cameras: Center Left Right
        source_path= line[i]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/'+ filename
        #print(current_path)
        image = cv2.imread(current_path)
        image_array = np.array(image)
        #print(image_array.shape)
        images.append(image)
        #print(line[3])
        correction = 0.2 # parameter to tune for Left and Right cameras
        if i == 0:
            measurement = float(line[3])
            measurements.append(measurement)
        elif i == 1: #Left
            measurement = float(line[3]) + correction
            measurements.append(measurement)
        else: #Right
            measurement = float(line[3]) - correction
            measurements.append(measurement)

#data augmentation
augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

'''
Pre-process the training data
'''
# X_train = np.array(images)
# y_train = np.array(measurements)
# #print(X_train.shape)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

'''
Define the Model Architecture
'''
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x/255.0, input_shape = (160,320,3)))
# Crop views (ROI)
model.add(Cropping2D(cropping=((70,25),(0,0))))
# NAVIDIA End-to-End CNN
model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

'''
Train the model
'''
# Defien tensorboard here
# todo: change the name when update the architecture of the model
#model_save_path = '1_model'
#model_save_path = '2_model_NAVIDIA'
#model_save_path = '3_model_NAVIDIA_DA' #data_augmentation
#model_save_path = '4_model_NAVIDIA_DA_ROI' #ROI
model_save_path = '5_model_NAVIDIA_DA_ROI_MultiCameras' #MultiCameras

tensorboard_dir = './tensorboard/'+ model_save_path
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

tensorboard = TensorBoard(log_dir = tensorboard_dir)

# Define Model save checkpoint here
checkpoint_dir = './checkpoint/'+ model_save_path +'/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'weights-{epoch:02d}-{val_loss:.2f}.h5')
modelsave = ModelCheckpoint(
                            filepath= checkpoint_path,
                            save_best_only=True,
                            verbose=1
                            )

model.compile(loss='mse', optimizer='adam')
model.fit(x=X_train,y= y_train,
          validation_split=0.1,
          shuffle=True,
          nb_epoch=7,
          callbacks=[modelsave, tensorboard]
          )



'''
Save the final model for submission
'''
model.save('model.h5')
