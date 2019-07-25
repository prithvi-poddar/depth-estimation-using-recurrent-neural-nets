# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:41:53 2019

@author: prith
"""

import csv

with open("test.csv", "w", newline='') as f:
    writer = csv.writer(f)
    count = 7000
    for i in range(7001):
        writer.writerow(['data/kitti_test/%d.png'%count, 'inpainted/%d.npy'%count])
        count+=1
        if count==8001:
            break
        