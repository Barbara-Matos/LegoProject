import cv2
import numpy as np
import imutils
import math

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

frame = cv2.imread("prof2.jpeg")               # prof3.jpeg
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
upper_black = np.array([180, 255, 30])

lower_gray = np.array([0, 0, 40])
upper_gray = np.array([180, 18, 177])
# lower_gray = np.array([0, 0, 20])
# upper_gray = np.array([0, 0, 113])
#
lower_gray1 = np.array([106, 43, 59])#70
upper_gray1 = np.array([117, 157, 175])#184
#
# lower_gray2 = np.array([104, 63, 53])   # 105
# upper_gray2 = np.array([114, 174, 167])

# lower_gray3 = np.array([104, 63, 53])
# upper_gray3 = np.array([114, 174, 167])

# lower_brown = np.array([2, 126, 0])
# upper_brown = np.array([25, 250, 150])
lower_brown = np.array([6, 0, 0])
upper_brown = np.array([83, 76, 198])
#lower_brown = np.array([4, 79, 50])
#upper_brown = np.array([32, 164, 140])
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
#mask62 = cv2.inRange(hsv, lower_gray2, upper_gray2)
maskGray = mask6 | mask61 #| mask62

mask7 = cv2.inRange(hsv, lower_black, upper_black)
mask71 = cv2.inRange(hsv, lower_black1, upper_black1)
maskBlack = mask7 | mask71

mask8 = cv2.inRange(hsv, lower_orange, upper_orange)
mask9 = cv2.inRange(hsv, lower_brown, upper_brown)

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

cnts9 = cv2.findContours(mask9, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts9= imutils.grab_contours(cnts9)

for c in cnts1:
    area1 = cv2.contourArea(c)
    if area1 > 10000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)  # o contorno sair melhor
        cv2.drawContours(frame, [approx], -1, (0,  0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                        approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        M = cv2.moments(c)
        print("DISTANCIASAMARELAS:",d)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Yellow", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4", (cx + 40, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts2:
    area2 = cv2.contourArea(c)
    if area2 > 7500:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0,  0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        print("distanciasverdes:" , d)
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
for c in reds:
    #for c in i:
    area3 = cv2.contourArea(c)
    if area3 > 30000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1 , (0 , 255 , 0) , 20)
        print("vertices: ",approx)
        #print("approx:",approx[i,0,0])
        i=0
        d=[None for _ in range(len(approx))]
        while(i< len(approx)):
            if i == (len(approx)-1):
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ])**2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ])**2)
            else:
                d[i]=math.sqrt((approx[i,0,0]-approx[i+1,0,0])**2 + (approx[i,0,1]-approx[i+1,0,1])**2 )
            i = i + 1

        print("distancias:", d)
        M = cv2.moments(c)

        cx = int(M[ "m10" ] / M[ "m00" ])
        cy = int(M[ "m01" ] / M[ "m00" ])
        cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
        cv2.putText(frame , "Red" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx) == 4 and (max(d) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts4:
    area4 = cv2.contourArea(c)
    if area4 > 7500:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                        approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        M = cv2.moments(c)
        print("azuis:",d)
        if len(approx) == 4:  # vertices
            print("retangulo azul")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.7) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.7):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts5:
    area5 = cv2.contourArea(c)
    if area5 > 7500:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                        approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Pink", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts6:
    area6 = cv2.contourArea(c)
    if area6 > 30000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                        approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Gray", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts7:
    area7 = cv2.contourArea(c)
    if area7 > 27000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                        approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Black", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts8:
    area8 = cv2.contourArea(c)
    if area8 > 27000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                        approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Orange", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame , "2X8" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)

for c in cnts9:
    area9 = cv2.contourArea(c)
    if area9 > 27000:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 20)
        i = 0
        d = [ None for _ in range(len(approx)) ]
        while (i < len(approx)):
            if i == (len(approx) - 1):
                d[ i ] = math.sqrt(
                    (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
            else:
                d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                        approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
            i = i + 1
        M = cv2.moments(c)
        print("brons:",d)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Brown", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
            cv2.putText(frame , "2X4", (cx +40, cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
            cv2.putText(frame , "2X3" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx)==4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
            cv2.putText(frame , "2X6" , (cx +40 , cy +40) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
        if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
            cv2.putText(frame, "2X8", (cx +40, cy +40), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

newframe = rescale_frame(frame, 18)      # MUDAR O VALOR PARA AJUSTAR O TAMANHO DA IMAGEM
cv2.imshow("result", newframe)
k = cv2.waitKey()
# if k == 27:
#   break
# cap.release()
cv2.destroyAllWindows()
