import cv2
import numpy as np
import imutils


def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

frame = cv2.imread("prof5.jpeg")               # imagem
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Gama de cores
lower_yellow = np.array([16, 68, 132])  # 120
upper_yellow = np.array([32, 255, 255])

lower_orange = np.array([5, 153, 80])  # 120
upper_orange = np.array([15, 255, 255])

lower_green = np.array([33, 64, 19])
upper_green = np.array([91, 255, 255])

lower_red = np.array([0, 50, 20])         # 120
upper_red = np.array([3, 255, 255])

lower_red1 = np.array([167, 118, 20])      # 120
upper_red1 = np.array([180, 255, 255])

lower_blue = np.array([104, 172, 100])
upper_blue = np.array([118, 255, 255])

lower_pink = np.array([134, 59, 71])
upper_pink = np.array([163, 255, 255])

lower_black1 = np.array([103, 0, 0])
upper_black1 = np.array([180, 255, 66])

lower_black = np.array([0, 0, 0])
upper_black = np.array([0, 0, 16])

lower_gray = np.array([0, 0, 20])
upper_gray = np.array([0, 0, 113])

lower_gray1 = np.array([104, 63, 72])
upper_gray1 = np.array([114, 100, 172])#184

lower_gray2 = np.array([104, 63, 53])   # 105
upper_gray2 = np.array([114, 174, 167])

#lower_gray3 = np.array([104, 63, 53])
#upper_gray3 = np.array([114, 174, 167])

# mascaras
mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask2 = cv2.inRange(hsv, lower_green, upper_green)

mask3 = cv2.inRange(hsv, lower_red, upper_red)
mask31 = cv2.inRange(hsv, lower_red1, upper_red1)
maskRed = mask3 | mask31

mask4 = cv2.inRange(hsv, lower_blue, upper_blue)
mask5 = cv2.inRange(hsv, lower_pink, upper_pink)

mask6 = cv2.inRange(hsv, lower_gray, upper_gray)
mask61 = cv2.inRange(hsv, lower_gray1, upper_gray1)
mask62 = cv2.inRange(hsv, lower_gray2, upper_gray2)
maskGray = mask61 | mask62 | mask6

mask7 = cv2.inRange(hsv, lower_black, upper_black)
mask71 = cv2.inRange(hsv, lower_black1, upper_black1)
maskBlack = mask7 | mask71

mask8 = cv2.inRange(hsv, lower_orange, upper_orange)

# find contourns
cnts1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts1 = imutils.grab_contours(cnts1)

cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = imutils.grab_contours(cnts2)

reds = cv2.findContours(maskRed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
reds = imutils.grab_contours(reds)

cnts4 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts4 = imutils.grab_contours(cnts4)

cnts5 = cv2.findContours(mask5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts5 = imutils.grab_contours(cnts5)

cnts6 = cv2.findContours(maskGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts6 = imutils.grab_contours(cnts6)

cnts7 = cv2.findContours(maskBlack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts7 = imutils.grab_contours(cnts7)

cnts8 = cv2.findContours(mask8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts8 = imutils.grab_contours(cnts8)

for c in cnts1:
    area1 = cv2.contourArea(c)
    if area1 > 10000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)  # o contorno sair melhor
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Yellow", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

for c in cnts2:
    area2 = cv2.contourArea(c)
    if area2 > 7500:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

for c in reds:
    #for c in i:
    area3 = cv2.contourArea(c)
    if area3 > 30000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1 , (0 , 255 , 0) , 3)
        M = cv2.moments(c)

        cx = int(M[ "m10" ] / M[ "m00" ])
        cy = int(M[ "m01" ] / M[ "m00" ])
        cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
        cv2.putText(frame , "Red" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts4:
    area4 = cv2.contourArea(c)
    if area4 > 7500:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        M = cv2.moments(c)

        if len(approx) == 4:  # vertices
            print("retangulo azul")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

for c in cnts5:
    area5 = cv2.contourArea(c)
    if area5 > 7500:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Pink", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

for c in cnts6:
    area6 = cv2.contourArea(c)
    if area6 > 30000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Gray", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

for c in cnts7:
    area7 = cv2.contourArea(c)
    if area7 > 27000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Black", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

for c in cnts8:
    area8 = cv2.contourArea(c)
    if area8 > 27000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Orange", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

newframe = rescale_frame(frame, 18)      # MUDAR O VALOR PARA AJUSTAR O TAMANHO DA IMAGEM
cv2.imshow("result", newframe)
k = cv2.waitKey()
# if k == 27:
#   break
# cap.release()
cv2.destroyAllWindows()
