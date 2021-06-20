import cv2
import numpy as np
import imutils
import math
import time
from imutils.video import VideoStream
from imutils.video import FPS
import argparse

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


TrDict = {'csrt': cv2.TrackerCSRT_create,
         'kcf' : cv2.TrackerKCF_create,
         'boosting' : cv2.TrackerBoosting_create,
         'mil': cv2.TrackerMIL_create,
         'tld': cv2.TrackerTLD_create,
         'medianflow': cv2.TrackerMedianFlow_create,
         'mosse':cv2.TrackerMOSSE_create}

trackers = cv2.MultiTracker_create()

v = cv2.VideoCapture('VideoTracking.mov')

ret, frame = v.read()

frame = rescale_frame(frame , 50)  # MUDAR O VALOR PARA AJUSTAR O TAMANHO DA IMAGEM
stringa=["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"]
color=["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"]
counter=0
k = 1
for i in range(k):
    cv2.imshow('Frame',frame)
    bbi = cv2.selectROI('Frame',frame, fromCenter=False,
                               showCrosshair=True)
    tracker_i = TrDict['mosse']()
    trackers.add(tracker_i,frame,bbi)

frameNumber = 2
#baseDir = r'D:\TrackingResults'

while True:
    ret, frame = v.read()
    frame = rescale_frame(frame , 50)

    
    
    if not ret:
        break
    (success,boxes) = trackers.update(frame)
    #np.savetxt(baseDir + '/frame_'+str(frameNumber)+'.txt',boxes,fmt='%f')
    frameNumber+=1
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        
        if x<2:
            m=1
        else: 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)   
            crop_img = frame[y:y+h, x:(x+w)] 

            lower_yellow = np.array([ 12 , 73 , 132 ])  # 120
            upper_yellow = np.array([ 32 , 255 , 255 ])


            lower_green = np.array([33, 64, 19])
            upper_green = np.array([91, 255, 255])


            lower_blue = np.array([104 , 172 , 100])
            upper_blue = np.array([121, 255 , 255])


            hsv = cv2.cvtColor(crop_img , cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv , lower_yellow , upper_yellow) 

            mask2 = cv2.inRange(hsv , lower_green , upper_green)

            mask4 = cv2.inRange(hsv , lower_blue , upper_blue)

            cnts1 = cv2.findContours(mask1 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
            cnts1 = imutils.grab_contours(cnts1)

            cnts2 = cv2.findContours(mask2 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
            cnts2 = imutils.grab_contours(cnts2)

            cnts4 = cv2.findContours(mask4 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
            cnts4 = imutils.grab_contours(cnts4)

            color[counter]='Color Undetected'
            for c in cnts1:
                area1 = cv2.contourArea(c)
                if area1 > 1000:
                
                    approx = cv2.approxPolyDP(c,0.02*cv2.arcLength(c,True),True) #o contorno sair melhor
                    color[counter]='yellow'
                    cv2.putText(frame , "Yellow" , (x,y)  , cv2.FONT_HERSHEY_SIMPLEX , 1, (255 , 255 , 255) , 1)

            for c in cnts2:
                area2 = cv2.contourArea(c)
                if area2 > 1000:
                    approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
                    #cv2.drawContours(crop_img , [ approx ] , -1 , (0 , 0 , 255) , 3)
                    color[counter]='green'
                    cv2.putText(frame , "Green" ,(x,y) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

            for c in cnts4:
                area4 = cv2.contourArea(c)
                if area4 > 1000:
                    approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
                    #cv2.drawContours(crop_img , [ approx ] , -1 , (0 , 0 , 255) , 3)
                    color[counter]='Blue'
                    cv2.putText(frame , "Blue" ,(x,y) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)

            


            if (w/h > 1.4 and w/h<2.3 and w*h>2500 and w*h<3900):
                cv2.putText(frame , '2x4' ,(x+(w-20),y+(h-20)) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
                stringa[counter]='2x4'

            if (w/h > 1.7 and w/h<2.3 and w*h>10500 and w*h<13200):
                cv2.putText(frame , "4x8" ,(x+(w-20),y+(h-20)) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
                stringa[counter]='4x8'

            if (w/h > 1.3 and w/h<1.8 and w*h>7800 and w*h<10000):
                cv2.putText(frame , "4x6" ,(x+(w-20),y+(h-20)) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
                stringa[counter]='4x6'

            if (w/h > 0.9 and w/h<1.1 and w*h>11000 and w*h<14500):
                cv2.putText(frame , "6x6" ,(x+(w-20),y+(h-20)) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
                stringa[counter]='6x6'
        #cv2.putText(frame , "Yellow" ,(x+w,y+h) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 1)
        #print("x:", x)
        #print("y:", y)
        #print("w:", w)
        #print("h:", h)
        #print("area: ",w*h)
         
        #cv2.imwrite('cropped.png', crop_img)
    
    ##cv2.imshow('crop',crop_img)

    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    

    if key == ord('q'):
        
        for y in range( len(boxes)):
            print("peace "+str(y)+": ", stringa[y]," ",color[y])
            
        break

    if key == ord("s"):
        counter+=1
        for i in range(k):
            cv2.imshow('Frame',frame)
            bbi = cv2.selectROI('Frame',frame, fromCenter=False,
                               showCrosshair=True)
            tracker_i = TrDict['mil']()
            trackers.add(tracker_i,frame,bbi)
    #print("tracker: ",len(boxes))
v.release()
cv2.destroyAllWindows()