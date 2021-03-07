#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:14:59 2019

@author: tom
"""
import time
import numpy as np
#import csv
#import math
from ctypes import c_bool
from multiprocessing import Process, Value
#import PIL
#from PIL import Image
import cv2 # type: ignore - pylance warning, since code runs on a different machine
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import sys
import jetson_camera

class ki_camera():
    def __init__(self):
        self.power_on_shm = Value(c_bool, True)
        self.tank_speed_shm = Value('d', 0)
        self.tank_direction_shm = Value('d', 0)
        self.mode_shm = Value('i', 0) # mode = (0, 1, 2) is (no action, record_train_data, predict direction) 
        self.run_camera_process = Process(target = self.run_camera, args =(self.power_on_shm, 
                                                                           self.mode_shm,
                                                                           self.tank_speed_shm,
                                                                           self.tank_direction_shm))
        self.run_camera_process.start()
    
    def load_ki_models(self):
        model_path = './model_follow_line_lego_01_trt_fp16'
        saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        return saved_model_loaded.signatures['serving_default']
        
    def run_camera(self, power_on_shm, mode_shm, tank_speed_shm, tank_direction_shm):
        infer_follow = self.load_ki_models()
        
        cap = jetson_camera.init_camera()
        turn = np.array([-1])


        while True:
            start_time = time.time()
            if (mode_shm.value == 1) and (tank_speed_shm.value > 0): # record mode, record only when driving (speed >0)
                image_arr = jetson_camera.get_image(cap)
                jetson_camera.log_drive_data(tank_speed_shm.value, tank_direction_shm.value, image_arr)
                time.sleep(0.1)
            elif mode_shm.value == 2: # auto mode, car is driving with AI-Power :-)
                arr = jetson_camera.get_image(cap)
                arr = arr / 255.0
                arr = arr.reshape(1,80,128,1)
                x = [arr, turn]
                x = arr.astype(np.float32)
                x = tf.constant(x)
                labeling = infer_follow(x)
                preds = labeling['reg_out'].numpy()
                dir_value = np.array(preds)[0]
                tank_direction_shm.value = dir_value
                print(dir_value, "  FPS:  ", int(1/(time.time() - start_time)))
                sys.stdout.flush()
            elif mode_shm.value ==3: # change direction
                turn = -1 * turn
                print("Turn Value = ", turn)
                mode_shm.value = 0
                print("Mode = ", mode_shm.value)
                sys.stdout.flush()
                
            if not(power_on_shm.value):
                break


if __name__ == "__main__":
    
    def test():
        run = 0
        my_ki_camera.mode_shm.value = 2 # mode 1 is record_train_data
        while True:
            run += 1
            if run > 8:
                my_ki_camera.power_on_shm.value = False
                my_ki_camera.run_camera_process.join()
                break
            if run > 1:
                my_ki_camera.tank_direction_shm.value = 0.33
                my_ki_camera.tank_speed_shm.value = -0.33
            time.sleep(10.9)
            
    
    my_ki_camera = ki_camera()
    time.sleep(3)
    test()
