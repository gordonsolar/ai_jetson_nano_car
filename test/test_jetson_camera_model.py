#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:14:59 2019

@author: tom
"""
#import io
#import socket
#import struct
import time
import numpy as np
#import csv
#import picamera
#import picamera.array
#from ctypes import c_bool
#import asyncio
#import uuid
#from multiprocessing import Process, Value
#import PIL
#from PIL import Image
import tensorflow as tf
#-import keras.preprocessing as prep
#-from keras.models import load_model
#import sys
import math
import cv2 # type: ignore - pylance warning, since code runs on a different machine

        
def gstreamer_pipeline(capture_width=640,capture_height=480,display_width=640,display_height=480,
                   framerate=10,flip_method=2,):
    return ("nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (capture_width,capture_height,framerate,flip_method,display_width,display_height,))


cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
model_path = './CNN_left_right.h5'       
model = tf.keras.models.load_model(model_path)

time.sleep(2)

if cap.isOpened(): 
    ret_val, imageHQ = cap.read()
    imageHQ = cv2.cvtColor(imageHQ, cv2.COLOR_BGR2GRAY)
    imageHQ = cv2.resize(imageHQ,(128,96),interpolation=cv2.INTER_AREA) # cv2.INTER_CUBIC cv2.INTER_LINEAR
    arr = np.array(imageHQ) / 255.0
    arr = arr[-60:,:]
    arr = arr.reshape(1,60,128,1)
    turn = np.array([1])
    pred = np.array(model.predict([arr, turn]))[0]
    dir_value = math.copysign(math.sqrt(math.fabs(pred)),pred) #(np.abs(max_direction * pred[0])+0.2)*signn
    if math.fabs(dir_value)>0.9: dir_value = math.copysign(1,dir_value)
    print(dir_value)
else: 
    print("Unable to open camera")

   