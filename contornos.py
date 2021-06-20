import cv2
import matplotlib
import notebook as notebook
import numpy as np
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv2.imread("prof7.png")               # imagem
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find contourns
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# NORMAL --------------------------------------------------- (FUNCIONA)
#for cnt in contours:
#    rect = cv2.minAreaRect(cnt)
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
#    img = cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

#plt.figure("Example 1")
#plt.imshow(img)
#plt.title('Contours in an image')
#plt.show()
#cv2.destroyAllWindows()

# Com aproximação ------------------------------------------- (FUNCIONA MELHOR DO QUE SEM A APROXIMAÇÃO)
for cnt in contours:
    epsilon = 0.01*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    img = cv2.drawContours(img, [approx], 0, (0, 255, 0), 5)

plt.figure("Example 2")
plt.imshow(img)
plt.title('Contours in an image com a aproximação')
plt.show()
cv2.destroyAllWindows()

# Momentos------------------------------------- (NÃO ESTÁ A FUNCIONAR CORRETAMENTE)
#for cnt in contours:
#    M = cv2.moments(cnt)

#    # Achar o centro do contour
#    if M["m00"] != 0:
#        cx = int(M['m10'] / M['m00'])
#        cy = int(M['m01'] / M['m00'])
#    else:
#        cx, cy = 0, 0

#    center = (cx, cy)

#    # Area
#    area = cv2.contourArea(cnt)

#    # Perímetro
#    perimeter = cv2.arcLength(cnt, True)

#    # A -> Area    P -> Perimeter
#    cv2.putText(img, "A: {0:2.1f}".format(area), center,
#    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 0, 0), 5)

#    cv2.putText(img, "P: {0:2.1f}".format(perimeter), (cx, cy +30),
#    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 0, 0), 3)

#plt.figure("Example 2")
#plt.imshow(img)
#plt.title('Contours in an image com momentos')
#plt.show()
#cv2.destroyAllWindows()