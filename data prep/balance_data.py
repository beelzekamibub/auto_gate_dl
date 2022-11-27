# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

final=[]
for i in range(1,13):
    train_data = np.load(f'data{i}.npy',allow_pickle=True)

    df = pd.DataFrame(train_data)
    print(df.head())
    print(Counter(df[1].apply(str)))

    lefts = []
    rights = []
    forwards = []
    backs=[]



    for data in train_data:
        img = data[0]
        choice = data[1]
    #0idx==left 1idx==forward 2idx==back 3idx==right    
        if choice == [1, 0, 0, 0]:
            lefts.append([img,choice])
        elif choice == [0, 1, 0, 0]:
            forwards.append([img,choice])
        elif choice == [0, 0 , 1, 0]:
            backs.append([img,choice])
        elif choice== [0, 0, 0, 1]:
            rights.append([img,choice])


    forwards = forwards[:len(lefts)][:len(rights)]
    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]
    final = final+ forwards + lefts + rights + backs
    print(f'forwards {len(forwards)}')
    print(f'backs {len(backs)}')
    print(f'lefts {len(lefts)}')
    print(f'rights {len(rights)}')
    print(len(final))
    
shuffle(final)

#np.save('upload_balanced.npy', final)




