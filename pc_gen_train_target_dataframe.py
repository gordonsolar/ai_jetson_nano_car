# %% import stuff
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# %% define Helper functions
def get_prep_image(fpath, recorded_image_shape, target_image_shape):
    image = tf.keras.preprocessing.image.load_img(fpath, color_mode="grayscale", 
                                                    target_size=(recorded_image_shape[0], recorded_image_shape[1])) #
    arr = tf.keras.preprocessing.image.img_to_array(image) / 255. # normalize RGB values to range [0,1]
    arr = arr[-1*target_image_shape[0]:,:,:]
    return arr

def gen_df_angle_speed_image(train_dir, train_df_filename, target_image_shape, debug, save_to_file):

    train_df = pd.read_csv(train_dir + 'train.csv', delimiter=',', names=['image_filename', 'steering_angle', 'acceleration'])
    train_df['turn'] = -1
    
    # %% Add image rgb values to training data
    train_df['image'] = train_df.apply(lambda x: get_prep_image(train_dir + x.image_filename, recorded_image_shape, image_shape), axis = 1)

    # check data frame after adding image data
    if debug: print(train_df.head())

    # save dataframe 
    if save_to_file: pickle.dump(train_df, file = open(train_dir + train_df_filename, "wb"))

    return train_df

# %% test module
if __name__ == "__main__":
    # define input parameter
    train_dir = '../training_data/' # Folder containing training data on local pc not tracked by git
    train_df_filename = 'train_dataframe.pkl'
    recorded_image_shape = (96, 128, 3) # Shape of images recorded by camera (width=128px, height=96px, 3 colors BGR)
    image_shape = (60,128,1)#(64,48,1) # Shape of images (width=64px, height=48px, 3 colors)
    debug = True
    save_to_file =  True
    gen_df_angle_speed_image(train_dir, train_df_filename, debug, save_to_file)

# %%
