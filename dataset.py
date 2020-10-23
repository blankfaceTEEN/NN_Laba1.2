import numpy as np
import cv2
import os

DIRPATH = 'train/'

files = os.listdir(DIRPATH)
images = []

for file_name in files:
    color_image = cv2.imread(DIRPATH + file_name)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    n_image = np.around(np.divide(gray_image, 255.0), decimals=1)
    images = np.append(images, n_image)

images = images.reshape(14, 100)
