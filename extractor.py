import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from PIL import Image
import matplotlib





dataset = pd.read_csv('data/kitti_test.csv')
rgb = dataset.iloc[:, 1]



count = 7000
for a in rgb:
    new = np.load(a)
    matplotlib.image.imsave('data/inpainted_new/%d.png'%count, new)
    count +=1
    
