import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
import tensorflow.keras.layers as tfl
from PIL import Image
import cv2
from collections import Counter
from random import shuffle

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception

from keras.models import Sequential
from tensorflow.keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import datetime
import tensorflow_hub as hub
import time
import gc
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from models import inception_v3 as googlenet
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode, mean
import numpy as np
from motion import motion_detection



GAME_WIDTH = 160
GAME_HEIGHT = 120

WIDTH = 120
HEIGHT = 160

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]


t_time = 0.25


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    # ReleaseKey(S)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():

    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
def create_model_url(model_url="https://tfhub.dev/tensorflow/efficientnet/b0/classification/1", num_classes=9):

    h, w, c = 120, 160, 1

    net = hub.KerasLayer(model_url, trainable=False)

    model = tf.keras.Sequential([
        Input(shape=(h, w, c)),
        Conv2D(3, (3, 3), padding='same'),
        net,
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    return model


#model = create_model_url()
#model.load_weights("C:\\Users\\chaitanya sharma\\OneDrive\\Desktop\\capstone\\new models\\effiecient_net\\efficient_model_checkpoint")
'''
model = tf.keras.models.Sequential([
    tfl.InputLayer(input_shape=(120,160,1)),

    tfl.Conv2D(filters=64,kernel_size=5,activation='relu'),
    tfl.MaxPool2D((2,2),2),
    tfl.Conv2D(filters=32,kernel_size=5,activation='relu'),
    tfl.MaxPool2D((2,2),2),
    tfl.Conv2D(filters=32,kernel_size=5,activation='relu'),
    tfl.MaxPool2D((2,2),2),
    tfl.Flatten(),

    tfl.Dense(1024,activation='relu'),
    tfl.Dropout(0.2),
    tfl.Dense(128,activation='relu'),
    tfl.Dropout(0.2),
    tfl.Dense(9,activation='softmax')
])
callback1 = tf.keras.callbacks.BackupAndRestore(backup_dir="/content/backup")
callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath="/content/model_checkpoint",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
callback3 = tf.keras.callbacks.TensorBoard(log_dir='/content/logs')
model.compile(
    optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07),
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)
model.load_weights("C:\\Users\\chaitanya sharma\\OneDrive\\Desktop\\capstone\\new models\\base testing\\checkpoint")'''

model = tf.keras.models.load_model(
    'C:\\Users\\chaitanya sharma\\OneDrive\\Desktop\\capstone\\new models\\base testing\\base_testing.h5')


print('We have loaded a previous model!!!!')

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    mode_choice = 0


    while(True):

        if not paused:
            screen = grab_screen(region=(0, 40, 800, 640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

            last_time = time.time()
            #screen = cv2.resize(screen, (120, 160))
            screen=np.array(screen)
            screen=screen.astype('float32').reshape(120,160)
            screen=screen/255
            prediction = model.predict(screen)[0]
            prediction = np.array(
                prediction) * np.array([4.5, 0.1, 0.1, 0.1,  1.8,   1.8, 0.5, 0.5, 0.2])

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'

            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'

            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
main()
