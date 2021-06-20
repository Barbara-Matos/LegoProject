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
    blur2 = cv2.bilateralFilter(frame , 20 , 30 ,30)  # se aplicarmos os filtros todos para processar a imagem obtemos melhores resultadoscv2.imshow("Blur2" , blur2)
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
        if area1 > 1000:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            # print("x:" , x)
            # print("y:", y)
            # print("w:", w)
            # print("h:", h)
            # print("area: ",w*h)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Yellow" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 1000:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Green" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)
    for c in reds:
        # for c in i:
        area3 = cv2.contourArea(c)
        if area3 > 500:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Red" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)
    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4 > 500:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Yellow" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)
    for c in cnts5:
        area5 = cv2.contourArea(c)
        if area5 > 1500:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Pink" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)
    for c in cnts6:
        area6 = cv2.contourArea(c)
        if area6 > 50000:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Gray" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

    for c in cnts10:
        area10 = cv2.contourArea(c)
        if area10 > 900:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Violet" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

    for c in cnts11:
        area11 = cv2.contourArea(c)
        if area11 > 1000:
            (x , y , w , h) = cv2.boundingRect(c)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 5)

            M = cv2.moments(c)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Beige" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 2.5 , (255 , 255 , 255) , 3)
            if (h / w > 1.4 and h / w < 2.3 and w * h > 2500 and w * h < 279000):
                cv2.putText(frame , '2x4' , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 3)

            if (w / h > 1.7 and w / h < 2.3 and w * h > 10500 and w * h < 13200):
                cv2.putText(frame , "4x8" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 1.3 and w / h < 1.8 and w * h > 7800 and w * h < 10000):
                cv2.putText(frame , "4x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

            if (w / h > 0.9 and w / h < 1.1 and w * h > 11000 and w * h < 14500):
                cv2.putText(frame , "6x6" , (x + (w - 20) , y + (h - 20)) , cv2.FONT_HERSHEY_SIMPLEX , 4 ,(255 , 255 , 255) , 1)

    cv2.imshow("result" , frame)
    k = cv2.waitKey(5)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()