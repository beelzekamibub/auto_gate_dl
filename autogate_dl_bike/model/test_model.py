from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(
    LR, 'alexnetv2', EPOCHS)
t_time = 0.07
forwards = 0


def straight():
    global forwards
    if(forwards > 4):
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        # time.sleep(0.2)
        forwards = 0
        return

    forwards = forwards+1
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    # time.sleep(0.03)
    # ReleaseKey(W)


def left():
    # PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)


def right():
    # PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)


def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):

        if not paused:

            screen = grab_screen(region=(0, 40, 800, 640))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160, 120))

            prediction = model.predict([screen.reshape(160, 120, 1)])[0]
            print(prediction)

            left_thresh = 0.65
            fwd_thresh = 0.70
            right_thresh = 0.95

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > left_thresh:
                left()
            elif prediction[2] > right_thresh:
                right()
            else:
                straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'Q' in keys:
            break
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
