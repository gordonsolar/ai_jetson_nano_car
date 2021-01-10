# %% import stuff
#import matplotlib.pyplot as plt
#from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# %% define functions

def gen_df_angle_speed_image(train_dir, train_df_filename, recorded_image_shape, target_image_shape, debug, save_to_file):

    def get_prep_image(fpath): #, recorded_image_shape, target_image_shape
        image = tf.keras.preprocessing.image.load_img(fpath, color_mode="grayscale", 
                                                        target_size=(recorded_image_shape[0], recorded_image_shape[1])) #
        arr = tf.keras.preprocessing.image.img_to_array(image) / 255. # normalize RGB values to range [0,1]
        arr = arr[-1*target_image_shape[0]:,:,:] # take only upper part of image without the part which shows the front car
        return arr

    train_df = pd.read_csv(train_dir + 'train.csv', delimiter=',', names=['image_filename', 'steering_angle', 'acceleration'])
    train_df['turn'] = -1
    
    # %% Add image rgb values to training data
    train_df['image'] = train_df.apply(lambda x: get_prep_image(train_dir + x.image_filename), axis = 1)

    # check data frame after adding image data
    if debug: print(train_df.head())

    # save dataframe 
    if save_to_file: pickle.dump(train_df, file = open(train_dir + train_df_filename, "wb"))

    return train_df

def gen_numpy_training():
    # %% convert data frames to numpy arrays suitable for model training (input and output) 
    # x_train are the input images 
    x_train = np.stack(train_df.image.values)
    # x_turn_train are the values for the turn direction 
    x_turn_train = train_df.turn.values
    # y_reg_train is the target steering angle obtained by regression model
    y_reg_train = train_df.steering_angle.values / max_y
    #y_class_train = train_df.track_in_view.values  # classification model for track_in_view information not used currently

    # %% Data augmentation: Supplement data with horizontally flipped images and corresponding inverted steering and turn value
    augment = False
    if augment:
        X = np.append(x_train, np.flip(x_train, 2), axis=0)
        X_turn = np.append(x_turn_train, -x_turn_train, axis=0)
        YR = np.append(y_reg_train, -y_reg_train, axis=0)
    else:
        X = x_train 
        X_turn = x_turn_train
        YR = y_reg_train 


# %% test module
if __name__ == "__main__":
    # define input parameter
    train_dir = '../training_data/' # Folder containing training data on local pc not tracked by git
    train_df_filename = 'train_dataframe.pkl'
    recorded_image_shape = (96, 128, 3) # Shape of images recorded by camera (width=128px, height=96px, 3 colors BGR)
    target_image_shape = (60,128,1)#(64,48,1) # Shape of images (width=64px, height=48px, 3 colors)
    train_df = gen_df_angle_speed_image(train_dir, train_df_filename,  recorded_image_shape, target_image_shape, debug = True, save_to_file = False)
