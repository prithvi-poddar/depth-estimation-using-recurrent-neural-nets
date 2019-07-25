import numpy as np
import cv2
import pandas as pd
import pickle

data_train = pd.read_csv('train.csv')
rgb = data_train.iloc[:,0]

train_rgb = []
load = []

count = 1
for a in rgb:
    img = cv2.imread(a, cv2.IMREAD_COLOR)
    load.append(img)
    count += 1
    if count == 33:
        count = 1
        train_rgb.append(load)
        load = []
with open('train_rgb.pickle', "wb") as f:
    pickle.dump(train_rgb, f)

depth = data_train.iloc[:,1]

train_depth = []
load = []

count = 1
for a in depth:
    img = np.transpose(np.load(a))
    load.append(img)
    count += 1
    if count == 33:
        count = 1
        train_depth.append(load)
        load = []
with open('train_depth.pickle', "wb") as f:
    pickle.dump(train_depth, f)


data_test = pd.read_csv('train.csv')
rgb = data_test.iloc[:,0]

test_rgb = []
load = []

count = 1
for a in rgb:
    img = cv2.imread(a, cv2.IMREAD_COLOR)
    load.append(img)
    count += 1
    if count == 33:
        count = 1
        test_rgb.append(load)
        load = []
with open('test_rgb.pickle', "wb") as f:
    pickle.dump(test_rgb, f)

depth = data_test.iloc[:,1]

test_depth = []
load = []

count = 1
for a in depth:
    img = np.transpose(np.load(a))
    load.append(img)
    count += 1
    if count == 33:
        count = 1
        test_depth.append(load)
        load = []
with open('test_depth.pickle', "wb") as f:
    pickle.dump(test_depth, f)