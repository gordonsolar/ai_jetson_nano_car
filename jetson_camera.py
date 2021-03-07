#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:14:59 2019

@author: tom
"""
import time
import numpy as np
import csv
#import math
#from ctypes import c_bool
#from multiprocessing import Process, Value
#import PIL
#from PIL import Image
import cv2 # type: ignore - pylance warning, since code runs on a different machine
#import tensorflow as tf
#from tensorflow.python.saved_model import tag_constants
#import sys


def gstreamer_pipeline(capture_width=640,capture_height=480,
                        display_width=128, display_height=96,
                        framerate=120, flip_method=0,):
    return ("nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
            % (capture_width,capture_height,framerate,flip_method,display_width,display_height,))
    
def init_camera():
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    time.sleep(2)
    return cap

def get_image(cap):
    if cap.isOpened(): 
        ret_val, imageHQ = cap.read()
        imageHQ = cv2.cvtColor(imageHQ, cv2.COLOR_BGR2GRAY)
        arr = np.array(imageHQ) #/ 255.0
        arr = arr[-96:-16,:]
        return arr
    else: 
        print("Unable to open camera")

def log_drive_data(speed, direction, image):
    image_fname = str(round(time.time()*100)) + ".png"
    cv2.imwrite("training_data/" + image_fname, image)
    direction_str = str(round(direction,3))
    speed_str = str(round(speed,3))
    data = [image_fname, direction_str, speed_str]
    with open(r"training_data/train.csv", "a") as f:
        wrt = csv.writer(f, lineterminator="\n")
        wrt.writerow(data)




if __name__ == "__main__":
    
    cap = init_camera()
    image = get_image(cap)
    cv2.imwrite("test_image.png", image)