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
from ctypes import c_bool
from multiprocessing import Process, Value
#import PIL
from PIL import Image
import cv2 # type: ignore - pylance warning, since code runs on a different machine
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import sys

class tank_camera(): #server_ip = '192.168.178.47', server_port = 8000
    def __init__(self):
        #self.power_on = True
        self.power_on_shm = Value(c_bool, True)
        #self.object_distance = 999
        # der ganze process kram ...
        self.tank_speed_shm = Value('d', 0)
        self.tank_direction_shm = Value('d', 0)
        self.mode_shm = Value('i', 0) # mode = (0, 1, 2) is (no action, record_train_data, predict direction) 
        self.run_camera_process = Process(target = self.run_camera, args =(self.power_on_shm, 
                                                                           self.mode_shm,
                                                                           self.tank_speed_shm,
                                                                           self.tank_direction_shm))
        self.run_camera_process.start()
        
    def gstreamer_pipeline(self,capture_width=1280, capture_height=720,
                            display_width=128, display_height=96,
                            framerate=120, flip_method=2,):
        return ("nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
                % (capture_width,capture_height,framerate,flip_method,display_width,display_height,))
        
        
    def run_camera(self, power_on_shm, mode_shm, tank_speed_shm, tank_direction_shm):
        #model_path = './CNN_follow_tf2.h5'
        model_path = './follow_saved_model_tf2_TFTRT_FP16_lego1'
        #model = tf.keras.models.load_model(model_path)
        saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        turn = np.array([-1])
        #image = np.zeros((48,64,3))
        #imageHQ = np.zeros((640,480,3)) #np.zeros((144,192,3))
        #low_zero = np.zeros((30,640,3))
        #high_zero = np.zeros((200,640,3))

        time.sleep(2)

        while True:
            start_time = time.time()
            if (mode_shm.value == 1) and (tank_speed_shm.value > 0): # record mode, record only when driving (speed >0)
                direction_str = str(round(tank_direction_shm.value,3))
                speed_str = str(round(tank_speed_shm.value,3))
                if cap.isOpened(): 
                    ret_val, image = cap.read()
                    PILimage = Image.fromarray(image) #before PIL.Image.fromarray...
                    image_fname = str(round(time.time()*100)) + ".jpg"
                    data = [image_fname, direction_str, speed_str]
                    with open(r"../training_data/train.csv", "a") as f:
                        wrt = csv.writer(f, lineterminator="\n")
                        wrt.writerow(data)
                    PILimage.save("../training_data/" + image_fname)
                else: 
                    print("Unable to open camera")
                
            elif mode_shm.value == 2: # auto mode, car is driving with AI-Power :-)
                if cap.isOpened(): 
                    ret_val, imageHQ = cap.read()
                    imageHQ = cv2.cvtColor(imageHQ, cv2.COLOR_BGR2GRAY)
                    #imageHQ = cv2.resize(imageHQ,(128,96),interpolation=cv2.INTER_AREA) # cv2.INTER_CUBIC cv2.INTER_LINEAR
                    arr = np.array(imageHQ) / 255.0
                    arr = arr[-66:-6,:]
                    arr = arr.reshape(1,60,128,1)
                    x = [arr, turn]
                    x = arr.astype(np.float32)
                    x = tf.constant(x)
                    #print('inter.structured_output: --------------')
                    #print(infer.structured_outputs)
                    labeling = infer(x)
                    #print('labeling: --------------')
                    #print(labeling)
                    preds = labeling['reg_out'].numpy()
                    dir_value = np.array(preds)[0]
                    #pred = np.array(model.predict(x))[0]
                    #dir_value = pred#math.copysign(math.sqrt(math.fabs(pred)),pred) #(np.abs(max_direction * pred[0])+0.2)*signn
                    #if math.fabs(dir_value)>0.99: dir_value = math.copysign(1,dir_value)
                    tank_direction_shm.value = dir_value
                    print(dir_value, "  FPS:  ", int(1/(time.time() - start_time)))
                    sys.stdout.flush()
                else: 
                    print("Unable to open camera")
                
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
        my_tank_camera.mode_shm.value = 2 # mode 1 is record_train_data
        while True:
            run += 1
            if run > 8:
                my_tank_camera.power_on_shm.value = False
                my_tank_camera.run_camera_process.join()
                break
            if run > 1:
                my_tank_camera.tank_direction_shm.value = 0.33
                my_tank_camera.tank_speed_shm.value = -0.33
            time.sleep(10.9)
            
    
    my_tank_camera = tank_camera()
    time.sleep(3)
    test()
