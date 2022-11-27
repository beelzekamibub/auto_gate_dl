#https://www.youtube.com/watch?v=F4y4YOpUcTQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a&index=12
import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os

def keys_to_output(keys):
    #[A,W,S,D]
    output=[0,0,0,0]

    if 'A' in keys:
        output[0]=1

    elif 'D' in keys:
        output[3]=1
        
    elif 'S' in keys:
        output[2]=1

    elif 'W' in keys:
        output[1]=1
        

    return output


file_name='data12.npy'

if os.path.isfile(file_name):
    print('file exists')
    training_data=list(np.load(file_name,allow_pickle=True))
else:
    print('making new')
    training_data=[]
    
timel=[]
def main():

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    last_time = time.time()
    paused = False
    while True:

        if not paused:
            screen = grab_screen(region=(0,40,800,640))
            screen= cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen=cv2.resize(screen,(160,120))
            last_time = time.time()
            keys=key_check()
            output=keys_to_output(keys)
            print(output)
            training_data.append([screen,output])
            timel.append(time.time()-last_time)
            #print('Frame took {} seconds'.format(time.time()-last_time))
            cv2.imshow('window2',screen)
            if len(training_data)%500==0:
                print('saving')
                np.save(file_name,training_data)
            
            if cv2.waitKey(25) & 0xFF==ord('q'):
                cv2.destroyAllWindows()
                break
        
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

main()
print(sum(timel)/len(timel))
