import os
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


# path for exported data, np array
DATA_PATH = os.path.join('MP_Data')
# Detecting actions
actions = np.array(['hello','thanks','salanghaeyo~','sad','family'])
# 30 videos of data
no_sequences = 30
# 30 frames
sequence_length = 30
# Folder start
start_folder = 0

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# put training data into array
label_map = {label:num for num, label in enumerate(actions)} # create dict for training data

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):  # looping through each vid
        window = []  # represent frames
        for frame_num in range(sequence_length): # looping through each frame
            res = np.load(os.path.join(DATA_PATH, action, str(sequence),  "{}.npy".format(frame_num)))  # passing frame to window
            window.append(res) 
        sequences.append(window) 
        labels.append(label_map[action])
x = np.array(sequences)
y = to_categorical(labels).astype(int)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)  # split into train and test data(test partition = 5% of train data)
#x_train.shape, x_test.shape


# build and train  model
log_dir = os.path.join('Logs') #create logs for trained model
tb_callback = TensorBoard(log_dir=log_dir)   #use tensorboard for data graph

model = Sequential() #linear stack layer -> 1 input 1 output
#  add layers to model
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) #shape of 30 frames prediction with 1662 keypoints
model.add(LSTM(128, return_sequences=True, activation='relu')) #activation == algorithm
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=80, callbacks=[tb_callback])


# save then del model
model.save('actions.h5')
del model

'''# EVALUATION
predictor = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
predictor = np.argmax(predictor,axis=1).tolist()

multilabel_confusion_matrix(ytrue, predictor)
accuracy_score(ytrue, predictor)

'''