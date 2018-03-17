'''
model.py containing the script to create and train the model
'''

import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense

from keras.callbacks import TensorBoard, ModelCheckpoint
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
    source_path= line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/'+ filename
    #print(current_path)
    image = cv2.imread(current_path)
    image_array = np.array(image)
    #print(image_array.shape)
    images.append(image)
    #print(line[3])
    measurement = float(line[3])
    measurements.append(measurement)

'''
Pre-process the training data
'''
X_train = np.array(images)
y_train = np.array(measurements)
#print(X_train.shape)

'''
Define the Model Architecture
'''

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

'''
Train the model
'''
# Defien tensorboard here
# todo: change the name when update the architecture of the model
model_save_path = '1_model'

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

