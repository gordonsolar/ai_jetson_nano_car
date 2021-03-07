#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:14:59 2019

@author: tom
"""
import time
import numpy as np
import cv2 # type: ignore - pylance warning, since code runs on a different machine

        
def gstreamer_pipeline(capture_width=640,capture_height=480,display_width=128,display_height=96,
                   framerate=100,flip_method=0,):
    return ("nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
        % (capture_width,capture_height,framerate,flip_method,display_width,display_height,))


cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

time.sleep(2)

if cap.isOpened(): 
    ret_val, imageHQ = cap.read()
    imageHQ = cv2.cvtColor(imageHQ, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("test_image.png", imageHQ)
    arr = np.array(imageHQ)# / 255.0
    arr = arr[-96:-16,:]
    cv2.imwrite("test_image_resize.png", arr)
else: 
    print("Unable to open camera")
    