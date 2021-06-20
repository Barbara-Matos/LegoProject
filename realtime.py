import cv2
import numpy as np
import imutils
import math

cap= cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
kernel = np.ones((5,5),np.uint8)

# Gama de cores
lower_yellow = np.array([ 12 , 73 , 132 ])  # 120
upper_yellow = np.array([ 32 , 255 , 255 ])

lower_orange = np.array([ 0 , 108 , 80 ])  # 120
upper_orange = np.array([ 9 , 255 , 255 ])

lower_green = np.array([ 48 , 36 , 19 ])
upper_green = np.array([ 102 , 255 , 158 ])

lower_red = np.array([ 0 , 134 , 20 ])  # 120
upper_red = np.array([ 3 , 255 , 255 ])

lower_red1 = np.array([ 167 , 118 , 20 ])  # 120
upper_red1 = np.array([ 180 , 255 , 255 ])

lower_blue = np.array([ 102 , 77 , 100 ])
upper_blue = np.array([ 118 , 255 , 255 ])

lower_violet = np.array([ 133 , 85 , 60 ])
upper_violet = np.array([ 159 , 255 , 255 ])

while True:
    _, frame = cap.read()
    
    hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)  # imagem

    # mascaras
    mask1 = cv2.inRange(hsv , lower_yellow , upper_yellow)
    mask1 = cv2.morphologyEx(mask1 , cv2.MORPH_OPEN , kernel)  # opening - erosion followed by dilation
    mask1 = cv2.morphologyEx(mask1 , cv2.MORPH_CLOSE , kernel)

    mask2 = cv2.inRange(hsv , lower_green , upper_green)
    mask2 = cv2.morphologyEx(mask2 , cv2.MORPH_OPEN , kernel)  # opening - erosion followed by dilation
    mask2 = cv2.morphologyEx(mask2 , cv2.MORPH_CLOSE , kernel)  # closing - dilation followed by erosion

    mask3 = cv2.inRange(hsv , lower_red , upper_red)
    mask31 = cv2.inRange(hsv , lower_red1 , upper_red1)
    maskRed = mask3 | mask31

    mask4 = cv2.inRange(hsv , lower_blue , upper_blue)
    mask8 = cv2.inRange(hsv , lower_orange , upper_orange)
    mask10 = cv2.inRange(hsv , lower_violet , upper_violet)
    #encontrar os contornos
    cnts1 = cv2.findContours(mask1 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    reds = cv2.findContours(maskRed , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    reds = imutils.grab_contours(reds)

    cnts4 = cv2.findContours(mask4 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)
    
    cnts8 = cv2.findContours(mask8 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts8 = imutils.grab_contours(cnts8)
    
    cnts10 = cv2.findContours(mask10 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    cnts10 = imutils.grab_contours(cnts10)
    
    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 > 800: #desenha um contorno 
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)  # o contorno sair melhor
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            
            M = cv2.moments(c)
            print("area amarela:" , area1)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Yellow" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            
            if (area1 >= 800 and  area1 <= 1300):
                cv2.putText(frame , "2X2" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area1 >= 1900 and  area1 <= 2200):
                cv2.putText(frame , "2X4" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area1 > 2200 and  area1 <= 2900):
                cv2.putText(frame , "2X5" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area1 >= 3000 and  area1 <= 3900):
                cv2.putText(frame , "2X6" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area1 > 3900 and  area1 <= 4900):
                cv2.putText(frame , "4x4" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area1 >= 5600 and  area1 <= 7700):
                cv2.putText(frame , "4X6" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
           
            if (area1 >= 8500 and  area1 <= 9500):
                cv2.putText(frame , "6X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 800:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            M = cv2.moments(c)
            #print("area2",area2)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Green" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area2 >= 800 and  area2 <= 1300):
                cv2.putText(frame , "2X2" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area2 >= 1900 and  area2 <= 2300):
                cv2.putText(frame , "2X4" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
           
            if (area2 > 2300 and  area2 <= 2900):
                cv2.putText(frame , "2X5" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area2 >= 3000 and  area2 <= 3899):
                cv2.putText(frame , "2X6" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area2 >= 3900 and  area2 <= 4900):
                cv2.putText(frame , "4x4" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area2 >= 5600 and  area2 <= 7700):
                cv2.putText(frame , "4X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area2 >= 8500 and  area2 <= 9500):
                cv2.putText(frame , "6X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
    for c in reds:
        area3 = cv2.contourArea(c)
        if area3 > 800:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 0) , 2)
            #print("Area_red:" , area3)
            M = cv2.moments(c)

            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Red" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            
            if (area3 >= 800 and  area3 <= 1300):
                cv2.putText(frame , "2X2" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area3 >= 1900 and  area3 <= 2200):
                cv2.putText(frame , "2X4" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area3 > 2200 and  area3 <= 2900):
                cv2.putText(frame , "2X5" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area3 >= 3000 and  area3 <= 3900):
                cv2.putText(frame , "2X6" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area3 >= 3900 and  area3 <= 4900):
                cv2.putText(frame , "4x4" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area3 >= 5600 and  area3 <= 7700):
                cv2.putText(frame , "4X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area3 >= 8500 and  area3 <= 9500):
                cv2.putText(frame , "6X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4 > 800:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            
            M = cv2.moments(c)
            #print("area retangulo azul",area4)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Blue" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            
            if (area4 >= 800 and  area4 <= 1300):
                cv2.putText(frame , "2X2" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area4 >= 1900 and  area4 <= 2200):
                cv2.putText(frame , "2X4" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area4 > 2200 and  area4 <= 2900):
                cv2.putText(frame , "2X5" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area4 >= 3000 and  area4 <= 3900):
                cv2.putText(frame , "2X6" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area4 >= 3900 and  area4 <= 4900):
                cv2.putText(frame , "4x4" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area4 >= 5600 and  area4 <= 7700):
                cv2.putText(frame , "4X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area4 >= 8500 and  area4 <= 9500):
                cv2.putText(frame , "6X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
    for c in cnts8:
        area8 = cv2.contourArea(c)
        if area8 > 800:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            
            M = cv2.moments(c)
            #print("azuis:" , d)
            
            print("area retangulo laranja",area8)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Orange" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            
            if (area8 >= 800 and  area8 <= 1300):
                cv2.putText(frame , "2X2" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area8 >= 1900 and  area8 <= 2200):
                cv2.putText(frame , "2X4" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area8 > 2200 and  area8 <= 2900):
                cv2.putText(frame , "2X5" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area8 >= 3000 and  area8 <= 3900):
                cv2.putText(frame , "2X6" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area8 >= 3900 and  area8 <= 4900):
                cv2.putText(frame , "4x4" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area8 >= 5600 and  area8 <= 7700):
                cv2.putText(frame , "4X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area8 >= 8500 and  area8 <= 9500):
                cv2.putText(frame , "6X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
    for c in cnts10:
        area10 = cv2.contourArea(c)
        if area10 > 800:
            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            cv2.drawContours(frame , [ approx ] , -1 , (0 , 0 , 255) , 2)
            
            M = cv2.moments(c)
            #print("azuis:" , d)
            
            print("area retangulo violeta",area10)
            cx = int(M[ "m10" ] / M[ "m00" ])
            cy = int(M[ "m01" ] / M[ "m00" ])
            
            cv2.circle(frame , (cx , cy) , 7 , (255 , 255 , 255) , -1)
            cv2.putText(frame , "Violet" , (cx - 20 , cy - 20) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
            
            if (area10 >= 800 and  area10 <= 1300):
                cv2.putText(frame , "2X2" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area10 >= 1900 and  area10 <= 2200):
                cv2.putText(frame , "2X4" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area10 > 2200 and  area10 <= 2900):
                cv2.putText(frame , "2X5" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
            
            if (area10 >= 3000 and  area10 <= 3900):
                cv2.putText(frame , "2X6" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area10 >= 3900 and  area10 <= 4900):
                cv2.putText(frame , "4x4" , (cx  , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area10 >= 5600 and  area10 <= 7700):
                cv2.putText(frame , "4X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
            if (area10 >= 8500 and  area10 <= 9500):
                cv2.putText(frame , "6X6" , (cx , cy + 40) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 1)
        
    cv2.imshow("result" , frame)
    k = cv2.waitKey(5)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()