import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
PATH = r"C:\Users\chait\Desktop\My Folder\Electronics Projects\Object Detection Using ESP32 CAM\object_samples\capsicum"
img_path = os.listdir(PATH)
result = np.empty([0,144])
for i in img_path:
  img = mpimg.imread(PATH + '/' + i)
  img = cv2.resize(img,dsize = (8,6), interpolation=cv2.INTER_CUBIC)
  img = img.flatten()
  result = np.vstack([result,img])
np.savetxt("pear4.csv",result, delimiter=",",fmt='%d')