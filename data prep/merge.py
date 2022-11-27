
import numpy as np
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle


final = []

for i in range(91, 110):
    train_data = np.load(
        f'C:\\Users\\chaitanya sharma\\OneDrive\\Desktop\\capstone\\data\\training_data-{i}.npy', allow_pickle=True)
    df = pd.DataFrame(train_data)

    w = []
    s = []
    a = []
    d = []
    wa = []
    wd = []
    sa = []
    sd = []
    nk = []

    for data in train_data:
        img = data[0]
        choice = data[1]

        if choice == [1, 0, 0, 0, 0, 0, 0, 0, 0]:
            w.append([img, choice])
        elif choice == [0, 1, 0, 0, 0, 0, 0, 0, 0]:
            s.append([img, choice])
        elif choice == [0, 0, 1, 0, 0, 0, 0, 0, 0]:
            a.append([img, choice])
        elif choice == [0, 0, 0, 1, 0, 0, 0, 0, 0]:
            d.append([img, choice])
        elif choice == [0, 0, 0, 0, 1, 0, 0, 0, 0]:
            wa.append([img, choice])
        elif choice == [0, 0, 0, 0, 0, 1, 0, 0, 0]:
            wd.append([img, choice])
        elif choice == [0, 0, 0, 0, 0, 0, 1, 0, 0]:
            sa.append([img, choice])
        elif choice == [0, 0, 0, 0, 0, 0, 0, 1, 0]:
            sd.append([img, choice])
        elif choice == [0, 0, 0, 0, 0, 0, 0, 0, 1]:
            nk.append([img, choice])

    final = final+w+s+a+d+wa+wd+sa+sd+nk
    print(len(final))

shuffle(final)

np.save('C:\\Users\\chaitanya sharma\\OneDrive\\Desktop\\capstone\\balanced data\\merged-10kbatch-28.npy', final)
