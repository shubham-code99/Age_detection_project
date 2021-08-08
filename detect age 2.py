import numpy as np
import cv2
import os

current_dir=os.getcwd()
os.chdir(current_dir+'\\data')
files=os.listdir()
data = []
dataLength = [0]
age = {}

for i in range(len(files)):
    userName = files[i].split('.')[0]
    age[i] = userName

for file in files:
    face = np.load(file)
    face = face.reshape(face.shape[0], -1)
    data.append(face)
    dataLength.append(len(face))

data = np.vstack(data)
labels = np.zeros((len(data), 1))
partition = len(files)

slice_1 = 0
slice_2 = 0

for i in range(len(dataLength) - 1):
    slice_1 += dataLength[i]
    slice_2 += dataLength[i + 1]
    labels[slice_1:slice_2 ] = int(i)

def dist(x2,x1):
    return np.sqrt(sum((x2 - x1) ** 2))

def knn(x,train):
    n = train.shape[0]
    distance = []
    for i in range(n):
        distance.append(dist(x,train[i]))
    distance = np.asarray(distance)
    indexes = np.argsort(distance)
    sortedLabels = labels[indexes][:5]
    count = np.unique(sortedLabels, return_counts=True)
    return count[0][np.argmax(count[1])]

os.chdir(current_dir)
font = cv2.FONT_HERSHEY_COMPLEX

face_dataset = cv2.CascadeClassifier('data.xml')
image = cv2.imread('young.jpg')
faces = face_dataset.detectMultiScale(image, 1.2)


while True:

    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        gray=cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        facess = gray[y:y + h, x:x + w]
        facess = cv2.resize(facess, (50, 50))
        label = knn(facess.flatten(), data)
        name = age[int(label)]
        cv2.putText(image, name, (x, y), font, 1, (255, 0, 0), 2)
    cv2.imshow('result',image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()