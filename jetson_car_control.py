#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:14:59 2019

@author: tom
"""
import asyncio
import jetson_gamepad_control
import jetson_motor_servo_control
import jetson_camera_control

async def remote_car_control():
    while True:
        if my_gamepad.button_x: #stop if x is pressed
            my_gamepad.power_on = False
            #my_gamepad.erase_rumble()
            my_movement_control.stop()
            my_tank_camera.power_on_shm.value = 0
            my_tank_camera.run_camera_process.join()
            break

        if my_gamepad.button_b: # toggle between record training data on and off
            my_gamepad.button_b = False
            if my_tank_camera.mode_shm.value != 1: 
                my_tank_camera.mode_shm.value = 1
                print("record mode on")
            else:
                my_tank_camera.mode_shm.value = 0
                print("record mode off")
        if my_gamepad.button_a: # toggle between auto mode on and off
            my_gamepad.button_a = False
            if my_tank_camera.mode_shm.value != 2: 
                my_tank_camera.mode_shm.value = 2
                print("auto mode on")
            else:
                my_tank_camera.mode_shm.value = 0
                print("auto mode off")
        if my_gamepad.button_y: # change turn direction
            my_gamepad.button_y = False
            my_tank_camera.mode_shm.value = 3 
            
        
        # in auto-ai-mode set direction to ai-prediction from camera and speed constant -0.4
        if my_tank_camera.mode_shm.value == 2: 
            if (my_gamepad.trigger_right > 1e-3) and (my_gamepad.trigger_left < 1e-3):
                speed = my_gamepad.trigger_right
            elif (my_gamepad.trigger_right < 1e-3) and (my_gamepad.trigger_left > 1e-3):
                speed = -1 * my_gamepad.trigger_left
            else:
                speed = 0
            my_tank_camera.tank_speed_shm.value = speed
        # in manual-mode set direction and speed with values from gamepad
        else:
            if (my_gamepad.trigger_right > 1e-3) and (my_gamepad.trigger_left < 1e-3):
                speed = my_gamepad.trigger_right
            elif (my_gamepad.trigger_right < 1e-3) and (my_gamepad.trigger_left > 1e-3):
                speed = -1 * my_gamepad.trigger_left
            else:
                speed = 0
            direction = my_gamepad.joystick_left_x
            my_tank_camera.tank_direction_shm.value = direction
            my_tank_camera.tank_speed_shm.value = speed
        # now update speed and direction for the movement control
        my_movement_control.set_speed_direction(my_tank_camera.tank_speed_shm.value, my_tank_camera.tank_direction_shm.value)
        await asyncio.sleep(0)
            

my_gamepad = jetson_gamepad_control.xbox_one(file = '/dev/input/event2')
my_movement_control = jetson_motor_servo_control.movement_control()
my_tank_camera = jetson_camera_control.tank_camera()

futures = [my_gamepad.read_gamepad_input(), remote_car_control()] #
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(futures))
loop.close()
