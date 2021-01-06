#!/usr/bin/python
# ============================================================================
# Raspi PCA9685 16-Channel PWM Servo Driver
# ============================================================================

import time
import math
import smbus


class PCA9685:
    '''PCA9685 Class'''
    #Registers/etc.
    __SUBADR1            = 0x02
    __SUBADR2            = 0x03
    __SUBADR3            = 0x04
    __MODE1              = 0x00
    __MODE2              = 0x01
    __PRESCALE           = 0xFE
    __LED0_ON_L          = 0x06
    __LED0_ON_H          = 0x07
    __LED0_OFF_L         = 0x08
    __LED0_OFF_H         = 0x09
    __ALLLED_ON_L        = 0xFA
    __ALLLED_ON_H        = 0xFB
    __ALLLED_OFF_L       = 0xFC
    __ALLLED_OFF_H       = 0xFD

    def __init__(self, address, debug=False):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        if self.debug:
            print("Reseting PCA9685")
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        "Writes an 8-bit value to the specified register/address"
        self.bus.write_byte_data(self.address, reg, value)
        if self.debug:
            print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))

    def read(self, reg):
        "Read an unsigned byte from the I2C device"
        result = self.bus.read_byte_data(self.address, reg)
        if self.debug:
            print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X"
                  % (self.address, result & 0xFF, reg))
        return result

    def set_pwm_freq(self, freq):
        "Sets the PWM frequency"
        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        if self.debug:
            print("Setting PWM frequency to %d Hz" % freq)
            print("Estimated pre-scale: %d" % prescaleval)
        prescale = math.floor(prescaleval + 0.5)
        if self.debug:
            print("Final pre-scale: %d" % prescale)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def set_pwm(self, channel, on, off):
        "Sets a single PWM channel"
        self.write(self.__LED0_ON_L + 4*channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4*channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4*channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4*channel, off >> 8)
        if self.debug:
            print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel,on,off))

    def set_dutycycle(self, channel, pulse):
        self.set_pwm(channel, 0, int(pulse * (4095 / 100)))

    def set_level(self, channel, value):
        if value == 1:
            self.set_pwm(channel, 0, 4095)
        else:
            self.set_pwm(channel, 0, 0)
   
    def stop_driver(self):
        self.write(self.__MODE2, 0x00)
        self.bus.close()

    def set_servo_pulse(self, channel, pulse):
        "Sets the Servo Pulse,The PWM frequency must be 50HZ"
        pulse = pulse*4096/20000        #PWM frequency is 50HZ,the period is 20000us
        self.set_pwm(channel, 0, int(pulse))

    def set_rotation_angle(self, channel, angle): 
        if(angle >= 28 and angle <= 135):
            temp = angle * (2000 / 180) + 525
            self.set_servo_pulse(channel, temp)
        else:
            print("Angle out of range")

class movement_control():
    def __init__(self):
        self.motor_driver = PCA9685(0x40, debug=False)
        self.motor_driver.set_pwm_freq(50)
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

    def set_speed_direction(self, speed, direction):
        speed = speed * -1
        direction = direction * -1
        if direction >= 0: # turn right, angle is from 85 to 130 
            self.motor_driver.set_rotation_angle(channel = 8, angle = 86 + (direction * 23))
        else: # turn left, angle is from 85 to 33
            self.motor_driver.set_rotation_angle(channel = 8, angle = 86 + (direction * 18))
        if speed >= 0:
            self.motor_driver.set_dutycycle(self.PWMB, speed * 100)
            self.motor_driver.set_level(self.BIN1, 1)
            self.motor_driver.set_level(self.BIN2, 0)
        else:
            self.motor_driver.set_dutycycle(self.PWMB, -1 * speed * 100)
            self.motor_driver.set_level(self.BIN1, 0)
            self.motor_driver.set_level(self.BIN2, 1)

    def stop(self):
        self.motor_driver.set_dutycycle(self.PWMA, 0)
        self.motor_driver.set_dutycycle(self.PWMB, 0)

#-------------------------------------------------------------------
if __name__ == "__main__":
    car_motor_control = movement_control()
    print("motor_test")
    car_motor_control.stop()
    car_motor_control.set_speed_direction(speed = 0.9, direction = -1)
    time.sleep(1)
    car_motor_control.set_speed_direction(speed = -0.9, direction = 1)
    time.sleep(1)
    car_motor_control.set_speed_direction(speed = 0., direction = 0)
    time.sleep(1)
    print("exit")
    car_motor_control.stop()

