import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

camera = cv.VideoCapture(0)
name = np.array(["f1.png","f2.png","f3.png","f4.png","f5.png","f6.png","f7.png","f8.png","f9.png","f10.png"])
cal = np.array(["c1.png","c2.png","c3.png","c4.png","c5.png","c6.png","c7.png","c8.png","c9.png","c10.png"])

#save images for calibration
'''
i = 0
while(i!=10):
    ret, img = camera.read()
    if ret == False:
        print("camara não detetada...")
    cv.imshow('Foto', img)
    if cv.waitKey(1) == ord("s"):
        cv.imwrite(name[i], img)
        i=i+1
camera.release()
'''
images = glob.glob('*.png')
xx=0
for i in range(len(name)):
    img = cv.imread(name[i])
    cv.imshow('Deteção', img)
    cv.waitKey(0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        img1=img
        cv.drawChessboardCorners(img, (7,7), corners2, ret)
        cv.imwrite(cal[xx], img)
        cv.imshow('img', img)
        cv.waitKey(0)
        xx=xx+1
        

#calibração
#retorna a matriz da camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Mxt: ", type(mtx))
print("Dist: ", type(dist))
print("Rotation vector: ", rvecs)
print("Translation Vector", tvecs)

#Save mtx and dist to a text file for the dataset.py script
#file = open("mtx","w")
#file.write(str(mtx))
#file.close()
np.save("mtx", mtx)
#file = open("dist","w")
#file.write(str(dist))
#file.close()
np.save("dist", dist)

#melhoramento da matriz
img = cv.imread('f8.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

#Guardar a matriz e o roi
np.save("matriz", newcameramtx)
np.save("roi",roi)

print("Matriz: ", newcameramtx)

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite("calibresult.png", dst)

#verificação do erro
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

cv.destroyAllWindows()
