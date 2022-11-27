import numpy as np
import scipy.ndimage
import cv2
import os
import time
import skimage.exposure
from numpy.random import default_rng
src='D:\\glitch directories\\train'
dest_base='D:\\glitch directories\\test'

c=1

classes=os.listdir(src)

for clas in classes:
    s=os.path.join(src,clas)
    d=os.path.join(dest_base,clas)
    images=os.listdir(s)
    for i in range(0,770):
        image_path=os.path.join(s,images[i])
        os.rename(image_path, os.path.join(d,images[i]))

