from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check
import numpy as np
import random
import tensorflow as tf
import tensorflow_hub as hub

t_time = 0.07
forwards = 0
w = [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
s = [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
a = [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
d = [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
wa = [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
wd = [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
sa = [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
sd = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
nk = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
keyw=[0.8, 0.1, 1.2, 1.2, 1.2, 1.2, 0.5, 0.5, 0.2]
counts=[0,0,0,0,0,0,0,0,0]
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def straight():
    global forwards
    if(forwards > 4):
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        #time.sleep(0.2)
        forwards = 0
        return

    forwards = forwards+1
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    # time.sleep(0.03)
    # ReleaseKey(W)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    # ReleaseKey(S))


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
    
path='C:\\Users\\chaitanya sharma\\OneDrive\\Desktop\\capstone project\\models\\base model extended batch data 10 epochs.h5'
#model = tf.keras.models.load_model((path),custom_objects={'KerasLayer':hub.KerasLayer})
model = tf.keras.models.load_model(path)
screen = grab_screen(region=(0, 40, 800, 640))

screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
screen = cv2.resize(screen, (160, 120))
screen=np.array(screen)
screen=screen.astype('float32')
screen=screen/255
prediction = model.predict([screen.reshape(1,120, 160, 1)])[0]
print(prediction)
def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):

        if not paused:

            screen = grab_screen(region=(0, 40, 800, 640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160, 120))
            screen=np.array(screen)
            screen=screen.astype('float32')
            screen=screen/255
            prediction = model.predict([screen.reshape(1,120, 160, 1)])[0]
            
            prediction = np.array(
                prediction) * np.array(keyw)
            mode_choice = np.argmax(prediction)
            print(prediction)
            if mode_choice == 0:
                counts[0]+=1
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                counts[1]+=1
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                counts[2]+=1
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                counts[3]+=1
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                counts[4]+=1
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                counts[5]+=1
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                counts[6]+=1
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                counts[7]+=1
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                counts[8]+=1
                no_keys()
                choice_picked = 'nokeys'
            print(choice_picked)

        keys = key_check()

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
print(counts)
