import cv2
import numpy as np
#import conveyor_lib
from tracker import *
import imutils

# Redefine image size
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Create tracker object
tracker = EuclideanDistTracker()


cap = cv2.VideoCapture("video1.mov")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

cntGreen = 0
cntYellow = 0
cntPink = 0
cntBlue = 0
cntOrange = 0
cntRed = 0
cntBlack = 0

while True:

    _, frame = cap.read()
    if frame is None:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract Region of interest
    roi = frame[550: 1450, 0: 2000]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 255, 255, cv2.THRESH_BINARY)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_frame, 90, 255, cv2.THRESH_BINARY)
    threshold[550:1450, 0:2000] = mask

    # Detect the Nuts
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    # Colors
    # Green --------------------------------------------------------------------
    lower_green = np.array([33, 64, 19])
    upper_green = np.array([91, 255, 255])
    green = cv2.inRange(hsv, lower_green, upper_green)
    cnt_green = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_green = imutils.grab_contours(cnt_green)

    # Yellow
    lower_yellow = np.array([16, 68, 132])
    upper_yellow = np.array([32, 255, 255])
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cnt_yellow = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_yellow = imutils.grab_contours(cnt_yellow)

    # Orange
    lower_orange = np.array([5, 153, 80])
    upper_orange = np.array([15, 255, 255])
    orange = cv2.inRange(hsv, lower_orange, upper_orange)
    cnt_orange = cv2.findContours(orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_orange = imutils.grab_contours(cnt_orange)

    # Blue
    lower_blue = np.array([104, 172, 100])
    upper_blue = np.array([118, 255, 255])
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    cnt_blue = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_blue = imutils.grab_contours(cnt_blue)

    # Pink
    lower_pink = np.array([134, 59, 71])
    upper_pink = np.array([163, 255, 255])
    pink = cv2.inRange(hsv, lower_pink, upper_pink)
    cnt_pink = cv2.findContours(pink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_pink = imutils.grab_contours(cnt_pink)

    # Red
    lower_red = np.array([0, 50, 20])  # 120
    upper_red = np.array([3, 255, 255])
    red = cv2.inRange(hsv, lower_red, upper_red)
    lower_red1 = np.array([167, 118, 20])  # 120
    upper_red1 = np.array([180, 255, 255])
    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    reds = red | red1
    cnt_red = cv2.findContours(reds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_red = imutils.grab_contours(cnt_red)

    # Black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([0, 0, 16])
    black = cv2.inRange(hsv, lower_black, upper_black)
    cnt_black = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_black = imutils.grab_contours(cnt_black)

    for cnt in cnt_green:
        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 2500 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    for cnt in cnt_yellow:
        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 2500 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Yellow", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    for cnt in cnt_orange:
        # Calculate area
        area = cv2.contourArea(cnt)

        AreaMax = 0
        if area > AreaMax:
            AreaMax = area

        # Distinguish small and big nuts
        if 2500 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Orange", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    for cnt in cnt_blue:
        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 2500 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    for cnt in cnt_pink:
        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 2500 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Pink", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    for cnt in cnt_red:
        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 20000 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Red", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    for cnt in cnt_black:
        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 2500 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Black", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

    newframe = rescale_frame(frame, 40)
    cv2.imshow("Frame", newframe)

    gray_newframe = rescale_frame(threshold, 20)
    cv2.imshow("Threshold", gray_newframe)

    newmask = rescale_frame(mask, 20)
    cv2.imshow("Mask", newmask)

    newroi = rescale_frame(roi, 20)
    cv2.imshow("roi", newroi)

    key = cv2.waitKey(1)
    if key == 27:
        break
print(cntGreen)
print(cntPink)
print(cntOrange)
print(cntYellow)
print(cntRed)

cap.release()
cv2.destroyAllWindows()