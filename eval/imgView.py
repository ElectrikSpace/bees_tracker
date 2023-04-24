import cv2
import numpy as np
import time
import skimage.measure
import matplotlib.pyplot as plt
import sys

img = cv2.imread(sys.argv[1], 0)
img2 = cv2.blur(img, (5, 5))
img3 = cv2.blur(img, (3, 3))
plt.imshow(img2)
plt.show()