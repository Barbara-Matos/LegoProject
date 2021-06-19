import cv2
import numpy as np
# import conveyor_lib
from tracker import *


# Tamanho da imagem
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("video1.mov")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=100)

while True:
    _, frame = cap.read()

    # Extract Region of interest
    roi = frame[800: 1200, 0: 2000]
    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 255, 255, cv2.THRESH_BINARY)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_frame, 90, 255, cv2.THRESH_BINARY)
    threshold[800:1200, 0:2000] = mask

    # Detect the Nuts
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 2500 < area < 55000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
            # Contorno vermelho

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

    newframe = rescale_frame(frame, 20)
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

cap.release()
cv2.destroyAllWindows()