import cv2
import matplotlib
import notebook as notebook
import numpy as np
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

img = cv2.imread("prof2.jpeg", cv2.IMREAD_GRAYSCALE) # imagem
_, threshold = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)    # Para ficar a preto e branco SÓ!

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    #cv2.drawContours(img, [cnt], 0, (0), 5)
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    img = cv2.drawContours(img, [approx], 0, (0, 255, 0), 5)

newImg1 = rescale_frame(img, 18)                # Nova escala da imagem
cv2.imshow("result", newImg1)

newImg2 = rescale_frame(threshold, 18)                # Nova escala da imagem
cv2.imshow("Threshold", newImg2)
cv2.waitKey(0)

cv2.destroyAllWindows()