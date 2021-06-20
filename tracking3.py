import cv2
from tracker import *

# Redefine image size
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Create tracker object
tracker = EuclideanDistTracker()

# Read de video
cap = cv2.VideoCapture("VideoTracking.mov")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    _, frame = cap.read()
    if frame is None:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract Region of interest
    roi = frame[100: 900, 0: 2000]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. Object Detection
    mask = object_detector.apply(gray_frame)
    _, mask = cv2.threshold(gray_frame, 254, 255, cv2.THRESH_BINARY)

    _, threshold = cv2.threshold(gray_frame, 90, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 20000 < area < 55000:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    #2. Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 5)

    newroi = rescale_frame(roi, 40)
    cv2.imshow("roi", newroi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()