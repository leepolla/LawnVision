
import cv2
import matplotlib as plt
import numpy
import os

img = cv2.imread("nobounds.png")

output = numpy.zeros((img.shape[0] ,img.shape[1], 3))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        #print(img[i][j])
        bgr = img[i][j]
        if (((bgr[1] > bgr[0]) and (bgr[1] > bgr[2])) and (bgr[1] > 30)):
            output[i][j] = [0, 50, 0]

cv2.imshow('test', output)
cv2.imshow('answer', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
