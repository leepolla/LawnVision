import cv2
import matplotlib as plt
import numpy as np
import os
#
# from skimage.measure import compare_ssim
# import argparse
# import imutils
# import cv2
#
# # construct the argument parse and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-f", "--first", required=True,
# #                 help="first input image")
# # ap.add_argument("-s", "--second", required=True,
# #                 help="second")
# # args = vars(ap.parse_args())
#
# # load the two input images
# imageA = cv2.imread("nobounds.png")
# imageB = cv2.imread(args["filledMap.png"])
#
# # convert the images to grayscale
# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
#
# # compute the Structural Similarity Index (SSIM) between the two
# # images, ensuring that the difference image is returned
# (score, diff) = compare_ssim(grayA, grayB, full=True)
# diff = (diff * 255).astype("uint8")
# print("SSIM: {}".format(score))
#
# # threshold the difference image, followed by finding contours to
# # obtain the regions of the two input images that differ
# thresh = cv2.threshold(diff, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# img = cv2.imread('lena.jpg')
# mask = cv2.imread('mask.png',0)
# res = cv2.bitwise_and(img,img,mask = mask)
image = cv2.imread("filledMap.png")
orig = cv2.imread("nobounds.png")
# image = np.zeros((400,400,3), dtype="uint8")


image[np.where((image != [255, 0, 0]).all(axis=2))] = [0, 0, 0]


#image[np.where((image != [0, 0, 0]).all(axis=2))] = [255, 255, 255]

#res = cv2.bitwise_and(orig,orig,mask = image)

cv2.imshow('test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
