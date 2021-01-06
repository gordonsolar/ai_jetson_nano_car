
# Introduction

The goal of this project is to realize an autonomous model car using deep learning (python + tensorflow).
Autonomous here means, that the car is able to follow a line, follow a path, turn left/right on a crossroad, avoid obstacles ...

In the current status of the project, the model car is able to follow a dashed line using a not so very deep neural network. Training data is obtained during manual remote steering of the car.

## Overview

The car is based on a custom lego technic chassis. The main components are shown in the picture.
![main components](https://github.com/gordonsolar/ai_jetson_nano_car/blob/master/pictures/ai_jetson_nano_car_lego.jpg)

## Electrical wiring

The wiring is pretty simple and compact using the [https://www.waveshare.com/motor-driver-hat.htm](waveshare-motor-driver-hat).
![Wiring diagram](https://github.com/gordonsolar/ai_jetson_nano_car/blob/master/pictures/wiring_diagram.png)

The motor-driver-hat can be attached directly to the jetson nano, allows to power the jetson and to control the motor.\
Using the free pin 15 of the IC PCA9685 it is also possible to control the steering servo. For Details see image and [https://www.waveshare.com/w/upload/8/81/Motor_Driver_HAT_User_Manual_EN.pdf](Manual).\
Powering the servo with the regulated 5V supply is typically not recommended - brownouts may occur. Here using a smaller servo (3kg) results in a stable system.

# Software

## System and tools setup

The jetson is running Ubuntu 18.04.5 LTS, Jetpack 4.4 and Tensoflow2.3.1 - standard image from NVidia.\
**Install editor**\
`sudo apt-get install nano`

**Configure Gui@Boot**\
`sudo systemctl set-default multi-user.target` to disable

`sudo systemctl set-default graphical.target`, `sudo systemctl start gdm3.service` to enable

**Install xbox controller support** (see also [https://github.com/atar-axis/xpadneo/issues/145](issue145)):\
`git clone https://github.com/atar-axis/xpadneo.git`\
`cd xpadneo`\
`sudo apt-get install dkms`\
`sudo ./install.sh`\
`sudo apt-get install sysfsutils`\
`sudo nano /etc/sysfs.conf`\
append _/module/bluetooth/parameters/disable_ertm=1_ to the end of the file and reboot

**Install python and tensorflow**\
see [https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html](Tensorflow&Nano)

**Install i2c / smbus** to control the waveshare motor-driver-hat\
`sudo apt-get install i2c-tools`\
`sudo i2cdetect -y -r 1` to check, if your hat is detected\
`sudo apt-get install python3-smbus`

**Install evdev** to read out the xbox controller\
`sudo pip3 install evdev`

## Car control software

To control the car, the main module `jetson_car_control.py`, which imports the modules `jetson_gamepad.py`, `jetson_motor_servo_control.py` and `jetson_camera_control.py` must be run on the jetson nano system.

The main module reads in the remote control commands from the Xbox controller via the module `jetson_gamepad.py` and allows to change between different modes: normal, auto-AI and record-training-data.

In normal mode it is possible to manually provide speed and direction.\
In auto-AI mode the steering angle is obtained from the module `jetson_camera_control.py` such that the car follows the dashed line on the floor.\
In record-training-data mode the manually provided steering angle and the camera view is recorded continuously to disk. The output consist of the recorded images and a CSV-file, which contains the images names with the corresponding steering angle information.

Direct control of the motor and the servo is accomplished through the module `jetson_motor_servo_control.py`.

## Model generation and training software

The data preparation, model generation and training is implemented in the script pc_gen_model_follow.py. It should be run on a pc (never tried on the jetson itself), ideally with tensorflow-gpu support. The script has the structure of a note book and runs from top to bottom.\
At the end the trained model is saved and subsequently converted into a TRT model, which is running faster on the "small" jetson nano Nvidia accelerator.
