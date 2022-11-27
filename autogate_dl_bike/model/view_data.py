import numpy as np
import cv2

train_data=np.load('C:\\Users\\chaitanya sharma\\OneDrive\\Desktop\\sentdex\\discrete data collections code\\data\\upload_balanced.npy',allow_pickle=True)

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))
'''
for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('window2',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF ==ord('q') :
        cv2.destroyAllWindows()
        break
    #print(data.shape)
        
i,n=train_data[200]
cv2.imshow('window2',cv2.cvtColor(i, cv2.COLOR_GRAY2RGB))
print(n)'''
