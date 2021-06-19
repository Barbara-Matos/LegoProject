import cv2
import numpy as np


# import conveyor_lib

# Tamanho da imagem
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture("video1.mov")

while True:
    _, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY)

    # Detect the Nuts
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Calculate area
        area = cv2.contourArea(cnt)

        # Distinguish small and big nuts
        if 5000 < area < 50000:
            # Contorno vermelho
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        # elif 100 < area < 400:
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        cv2.putText(frame, str(area), (x, y), 1, 1, (0, 255, 0))

    newframe = rescale_frame(frame, 25)
    cv2.imshow("Frame", newframe)

    gray_newframe = rescale_frame(threshold, 25)
    cv2.imshow("Threshold", gray_newframe)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()