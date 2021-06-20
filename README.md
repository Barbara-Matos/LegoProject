# Lego Project

This project consists of using a Raspberry Pi 4 Model B along with a camera in order to detect Lego pieces inside one box with controlled lightning. 

It will also exist a tracking feature, but this one it will only be aplied to the video called 'VideoTracking.mov'.



## Table of Contents
1. [Installation requires libraries](#reqlib)  
2. [Real time object detection of Lego pieces](#objectdetectionRT)
3. [Object tracking on video](#objecttracking)

<a name="reqlib"></a> 
## Installation Required Libraries

This guide will only be usefull if you are using a debian based operating system, if you use any other type of OS you will need to search for this information elsewhere.

In order to run the project on your own device you need to follow the following steps:

- Clone our repository to your computer
```
$ git clone https://github.com/Barbara-Matos/LegoProject.git
```
- Open your terminal and go to the directory where you clone the repository
- Run the following commands on your terminal
```
$ sudo apt-get install python3-opencv
$ pip3 install numpy
$ pip3 install imutils
$ pip3 install math
$ pip3 install glob
$ pip3 install time
$ pip3 install argparse
```
- Now run the following command replacing "program-name.py" with the name of the program you want to start running
```
$ python3 program-name.py
```

<a name="objectdetectionRT"></a>
## Real Time Object Detection of Lego Pieces 
###### realtime.py
This program recognizes lego pieces by color and shape.

The pieces have to have this caracteristics in order to our program to detect them.

Shapes: 2x2 - 2x4 - 2x5 - 2x6 - 4x4 - 4x6 

Colors: Yellow - Orange - Green - Red - Blue

<a name="objecttracking"></a>
## Object Tracking on Video

There are three diferent programs that have as objective tracking lego pieces on a video. 

###### tracking1.py
On the first one, tracking1.py, you need to press S button on keyboard in order to stop the video, then you select the region of interest (ROI) with your mouse and after that you press ENTER to resume the video, the program does the tracking of the selected region. You can make this process as many times as you want along the video for detect varied pieces.   

###### tracking2.py
On the second one, tracking2.py, the program detects the lego pieces by colour and does the tracking of them. In this program theres an issue with the identifiers of the pieces that makes impossible to count them correctly.  

###### tracking3.py
On the third one, tracking3.py, in this program it's chosen a small ROI where the light conditions doesn't interfere and the tracking is made on that region. The pieces' identifiers are applied correctly, but there isn't color detection involved on the process. 


