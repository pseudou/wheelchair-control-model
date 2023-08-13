
# making dataset

import glob
import numpy as np
import cv2

# import dataset
# importing down images
files = glob.glob("D:\\Pseudou\\studies\\8th sem\\eye_processing\\correct_dataset\\training\\down\\*.jpg")

dataset = []
data_label = []

m,n = 128,128

for file in files:
    
    img = cv2.imread(file)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img,(m,n))
    
    
    dataset.append(img)
    
    # 0 --> down
    data_label.append(0)
    

# importing left
files = glob.glob("D:\\Pseudou\\studies\\8th sem\\eye_processing\\correct_dataset\\training\\left\\*.jpg")

for file in files:
    
    img = cv2.imread(file)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img,(m,n))
        
    dataset.append(img)
    
    # 1 --> left
    data_label.append(1)


# importing right
files = glob.glob("D:\\Pseudou\\studies\\8th sem\\eye_processing\\correct_dataset\\training\\right\\*.jpg")

for file in files:
    
    img = cv2.imread(file)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img,(m,n))
       
    dataset.append(img)
    
    # 2 --> right
    data_label.append(2)


# importing up
files = glob.glob("D:\\Pseudou\\studies\\8th sem\\eye_processing\\correct_dataset\\training\\up\\*.jpg")

for file in files:
    
    img = cv2.imread(file)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img,(m,n))
    
    dataset.append(img)
    
    # 3 --> up
    data_label.append(3)
    

# importing straight
files = glob.glob("D:\\Pseudou\\studies\\8th sem\\eye_processing\\correct_dataset\\training\\straight\\*.jpg")

for file in files:
    
    img = cv2.imread(file)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img,(m,n))
    
    dataset.append(img)
    
    # 4 --> straight
    data_label.append(4)



dataset = np.array(dataset)
data_label = np.array(data_label)

np.save("D:\\Pseudou\\studies\\8th sem\\eye_processing\\correct_dataset\\dataset_training.npy",dataset)
np.save("D:\\Pseudou\\studies\\8th sem\\eye_processing\\correct_dataset\\data_label_training.npy",data_label)
