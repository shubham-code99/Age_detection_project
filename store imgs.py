import numpy as np
import pandas as pd
import os
import cv2

dataset = pd.read_csv('train.csv')

current_dir = os.getcwd()
os.chdir(current_dir+'\\train')
imgs = os.listdir()

df = pd.DataFrame(dataset)

middle = []
young = []
old = []

for i in range(len(df)):
    img = cv2.imread(df.iloc[i,0])
    face = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face , (50,50))
    if df.iloc[i,1] == 'MIDDLE':
            middle.append(face)
    elif df.iloc[i,1] == 'YOUNG':
            young.append(face)
    elif df.iloc[i,1] == 'OLD':
            old.append(face)

middle = np.asarray(middle)
young = np.asarray(young)
old = np.asarray(old)
os.chdir(current_dir)
np.save('data/middle.npy',middle)
np.save('data/young.npy',young)
np.save('data/old.npy',old)