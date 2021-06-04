import cv2
import numpy as np
import imutils
import math

#cap = cv2.VideoCapture('http://192.168.1.144:8080/video')
cap= cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
kernel = np.ones((5,5),np.uint8)
while True:
    _, frame = cap.read()
    blur = cv2.blur(frame , (5 , 5))  # Blur Filter
    blur0 = cv2.medianBlur(blur , 5)  # Median Filter
    blur1 = cv2.GaussianBlur(frame , (5 , 5) , 0)  # Gaussian Filter
    blur2 = cv2.bilateralFilter(frame , 20 , 30 ,
                                30)  # se aplicarmos os filtros todos para processar a imagem obtemos melhores resultados
    cv2.imshow("Blur2" , blur2)
    cv2.waitKey(0)
    hsv = cv2.cvtColor(blur2 , cv2.COLOR_BGR2HSV)  # imagem

    # Gama de cores
    lower_yellow = np.array([ 12 , 73 , 132 ])  # 120
    upper_yellow = np.array([ 32 , 255 , 255 ])

    lower_orange = np.array([ 0 , 108 , 80 ])  # 120
    upper_orange = np.array([ 9 , 255 , 255 ])

    # lower_green = np.array([33, 64, 19])
    # upper_green = np.array([91, 255, 255])
    lower_green = np.array([ 48 , 36 , 19 ])
    upper_green = np.array([ 102 , 255 , 158 ])

    lower_red = np.array([ 0 , 134 , 20 ])  # 120
    upper_red = np.array([ 3 , 255 , 255 ])

    lower_red1 = np.array([ 167 , 118 , 20 ])  # 120
    upper_red1 = np.array([ 180 , 255 , 255 ])

    lower_blue = np.array([ 102 , 77 , 100 ])
    upper_blue = np.array([ 118 , 255 , 255 ])

    lower_pink = np.array([ 134 , 59 , 71 ])
    upper_pink = np.array([ 163 , 255 , 255 ])

    lower_black1 = np.array([ 103 , 0 , 0 ])
    upper_black1 = np.array([ 180 , 255 , 66 ])

    lower_black = np.array([ 0 , 0 , 0 ])
    upper_black = np.array([ 180 , 255 , 30 ])

    lower_gray = np.array([ 0 , 0 , 40 ])
    upper_gray = np.array([ 180 , 18 , 177 ])
    # lower_gray = np.array([0, 0, 20])
    # upper_gray = np.array([0, 0, 113])
    #
    lower_gray1 = np.array([ 106 , 43 , 59 ])  # 70
    upper_gray1 = np.array([ 117 , 157 , 175 ])  # 184
    #
    # lower_gray2 = np.array([104, 63, 53])   # 105
    # upper_gray2 = np.array([114, 174, 167])

    # lower_gray3 = np.array([104, 63, 53])
    # upper_gray3 = np.array([114, 174, 167])

    lower_brown1 = np.array([ 0 , 52 , 52 ])
    upper_brown1 = np.array([ 6 , 135 , 86 ])

    lower_brown = np.array([ 149 , 65 , 0 ])
    upper_brown = np.array([ 180 , 172 , 101 ])

    lower_beige = np.array([ 0 , 33 , 107 ])
    upper_beige = np.array([ 17 , 82 , 190 ])

    lower_violet = np.array([ 133 , 85 , 60 ])
    upper_violet = np.array([ 159 , 255 , 255 ])
    # mascaras
    mask1 = cv2.inRange(hsv , lower_yellow , upper_yellow)
    # mask1 = cv2.erode(mask1,kernel,iterations = 1)
    mask1 = cv2.morphologyEx(mask1 , cv2.MORPH_OPEN , kernel)  # opening - erosion followed by dilation
    mask1 = cv2.morphologyEx(mask1 , cv2.MORPH_CLOSE , kernel)

    mask2 = cv2.inRange(hsv , lower_green , upper_green)
    mask2 = cv2.morphologyEx(mask2 , cv2.MORPH_OPEN , kernel)  # opening - erosion followed by dilation
    mask2 = cv2.morphologyEx(mask2 , cv2.MORPH_CLOSE , kernel)  # closing - dilation followed by erosion

    mask3 = cv2.inRange(hsv , lower_red , upper_red)
    mask31 = cv2.inRange(hsv , lower_red1 , upper_red1)
    maskRed = mask3 | mask31

    mask4 = cv2.inRange(hsv , lower_blue , upper_blue)
    mask5 = cv2.inRange(hsv , lower_pink , upper_pink)

    mask6 = cv2.inRange(hsv , lower_gray , upper_gray)
    mask61 = cv2.inRange(hsv , lower_gray1 , upper_gray1)
    # mask62 = cv2.inRange(hsv, lower_gray2, upper_gray2)
    maskGray = mask6 | mask61  # | mask62

    mask7 = cv2.inRange(hsv , lower_black , upper_black)
    mask71 = cv2.inRange(hsv , lower_black1 , upper_black1)
    maskBlack = mask7 | mask71

    mask8 = cv2.inRange(hsv , lower_orange , upper_orange)

    mask9 = cv2.inRange(hsv , lower_brown , upper_brown)
    mask91 = cv2.inRange(hsv , lower_brown1 , upper_brown1)
    mask9 = mask9 | mask91

    mask10 = cv2.inRange(hsv , lower_violet , upper_violet)

    mask11 = cv2.inRange(hsv , lower_beige , upper_beige)

    # find contourns
    cnts1 = cv2.findContours(mask1 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    reds = cv2.findContours(maskRed , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    reds = imutils.grab_contours(reds)

    cnts4 = cv2.findContours(mask4 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)

    cnts5 = cv2.findContours(mask5 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts5 = imutils.grab_contours(cnts5)

    cnts6 = cv2.findContours(maskGray , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts6 = imutils.grab_contours(cnts6)

    cnts7 = cv2.findContours(maskBlack , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts7 = imutils.grab_contours(cnts7)

    cnts8 = cv2.findContours(mask8 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts8 = imutils.grab_contours(cnts8)

    cnts9 = cv2.findContours(mask9 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts9 = imutils.grab_contours(cnts9)

    cnts10 = cv2.findContours(mask10 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts10 = imutils.grab_contours(cnts10)

    cnts11 = cv2.findContours(mask11 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts11 = imutils.grab_contours(cnts11)

    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 > 900:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)  # o contorno sair melhor
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)
            print("DISTANCIASAMARELAS:" , d)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Yellow" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 1000:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 3)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            print("distanciasverdes:" , d)
            M = cv2.moments(c)

            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Green" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
    for c in reds:
        # for c in i:
        area3 = cv2.contourArea(c)
        if area3 > 500:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 255 , 0) , 1)
            print("vertices: " , approx)
            # print("approx:",approx[i,0,0])
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                                approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1

            print("distancias:" , d)
            M = cv2.moments(c)

            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Red" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4 > 500:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)
            print("azuis:" , d)
            if len(approx) == 4:  # vertices
                print("retangulo azul")
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Blue" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts5:
        area5 = cv2.contourArea(c)
        if area5 > 1500:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 20)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)

            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Pink" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts6:
        area6 = cv2.contourArea(c)
        if area6 > 50000:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)

            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Gray" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts7:
        area7 = cv2.contourArea(c)
        if area7 > 27000:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 20)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)

            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Black" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts8:
        area8 = cv2.contourArea(c)
        if area8 > 1000:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)

            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Orange" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts9:
        area9 = cv2.contourArea(c)
        if area9 > 1000:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)
            print("browns:" , d)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Brown" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
    for c in cnts10:
        area10 = cv2.contourArea(c)
        if area10 > 900:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)  # o contorno sair melhor
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)
            print("DISTANCIASAMARELAS:" , d)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Violet" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    for c in cnts11:
        area11 = cv2.contourArea(c)
        if area11 > 1000:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)  # o contorno sair melhor
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            i = 0
            d = [ None for _ in range(len(approx)) ]
            while (i < len(approx)):
                if i == (len(approx) - 1):
                    d[ i ] = math.sqrt(
                        (approx[ i , 0 , 0 ] - approx[ 0 , 0 , 0 ]) ** 2 + (
                                    approx[ i , 0 , 1 ] - approx[ 0 , 0 , 1 ]) ** 2)
                else:
                    d[ i ] = math.sqrt((approx[ i , 0 , 0 ] - approx[ i + 1 , 0 , 0 ]) ** 2 + (
                            approx[ i , 0 , 1 ] - approx[ i + 1 , 0 , 1 ]) ** 2)
                i = i + 1
            M = cv2.moments(c)
            print("DISTANCIASAMARELAS:" , d)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Beige" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 2.2) and (max((d)) / min((d)) >= 1.8):
                cv2.putText(frame , "2X4" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 1.6) and (max((d)) / min((d)) >= 1.4):
                cv2.putText(frame , "2X3" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 3.2) and (max((d)) / min((d)) >= 2.8):
                cv2.putText(frame , "2X6" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            if len(approx) == 4 and (max((d)) / min((d)) <= 4.2) and (max((d)) / min((d)) >= 3.8):
                cv2.putText(frame , "2X8" , (cx + 40 , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

    cv2.imshow("result" , frame)
    k = cv2.waitKey(5)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
