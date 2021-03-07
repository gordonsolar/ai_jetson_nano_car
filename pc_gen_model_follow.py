# %% [markdown]
# # Autonomous Vehicle Modell
# ### -- CNN Architecture to follow a path

# %%
from __future__ import absolute_import, division, print_function, unicode_literals
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# %% -- import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import cv2

# %% -- define Helper functions
def get_prep_image(fpath):
    arr = np.array(cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)) / 255.0
    #image = tf.keras.preprocessing.image.load_img(fpath, color_mode="grayscale", target_size=(rescale_image_shape[0], rescale_image_shape[1])) #
    #arr = tf.keras.preprocessing.image.img_to_array(image) / 255. # normalize RGB values to range [0,1]
    #arr = arr[-1*image_shape[0]:,:,:]
    #arr=arr.reshape(arr.shape[0],arr.shape[1])
    return arr

def shuffle_arrays(arrays, set_seed=-1):
    """Shuffles arrays in-place, in the same order, along axis=0
    Necessary to shuffle training and validation data
    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

# %% -- define constants
train_dir = '../training_data/' # Folder containing training data with images and cvs file with steering / speed data
image_shape = (80,128,1) # Shape of images (height, width, colors)
max_y = 1 # Maximum steering angle
debug = True # print extra information during script run
augment = False #augment training data (flip images and steering angle data)?
model_name = '/home/tom/jetson_nano/ai_jetson_nano_car/model_follow_line_lego_01' # where model is saved
model_name_trt = 'model_follow_line_lego_01_trt_fp16' # where the reduced trt model is saved

# %% -- Label training data
# -- Load training data into Pandas data frame
train_df_left = pd.read_csv('../training_data/train.csv', delimiter=',', names=['image_filename', 'steering_angle', 'acceleration'])
train_df_left['turn'] = -1
#train_df_right = pd.read_csv('my_raspi_drives_AI/training_data_right/train.csv', delimiter=',', names=['image_filename', 'steering_angle', 'acceleration'])
#train_df_right['turn'] = 1

# -- Add image rgb values to training data
train_df_left['image'] = train_df_left.apply(lambda x: get_prep_image(train_dir + x.image_filename), 1)
#train_df_right['image'] = train_df_right.apply(lambda x: get_prep_image(train_dir_right + x.image_filename), 1)
train_df = train_df_left # 
#train_df = pd.concat([train_df_left, train_df_right], ignore_index=True)

if debug: print(train_df.head()) # check data frame after adding image data

# %% -- convert data frames to numpy arrays suitable for model training (input and output) 
x_train = np.stack(train_df.image.values) # -- x_train are the input images
x_turn_train = train_df.turn.values # -- x_turn_train are the values for the turn direction 
# -- y_reg_train is the target steering angle obtained by regression model
y_reg_train = train_df.steering_angle.values / max_y
#y_class_train = train_df.track_in_view.values  # classification model for track_in_view information not used currently

# %% -- Data augmentation: Supplement data with horizontally flipped images and corresponding inverted steering and turn value
if augment:
    X = np.append(x_train, np.flip(x_train, 2), axis=0)
    X_turn = np.append(x_turn_train, -x_turn_train, axis=0)
    YR = np.append(y_reg_train, -y_reg_train, axis=0)
else:
    X = x_train 
    X_turn = x_turn_train
    YR = y_reg_train 

#YC = np.append(y_class_train, y_class_train, axis=0) # classification model for track_in_view information not used currently
# now shuffle the data, so that the validation split contains samples from the complete set 
shuffle_arrays([X, X_turn, YR], set_seed=7)

# %% Preview data
if debug:
    fig=plt.figure(figsize=(16, 8))
    columns = 2
    rows = 2
    offs = np.random.randint(1,X.shape[0]-10)
    for i in range(1, columns*rows +1):
        img = np.random.randint(10, size=(25,25))
        fig.add_subplot(rows, columns, i)
        imi = X[offs + i]
        #imi = imi.reshape(imi.shape[0],imi.shape[1])
        plt.imshow(imi)
        plt.title('steering: ' + str(round(-YR[offs + i]*45,0)) + '°')
    plt.tight_layout()
    plt.show()

# %% Define Neural Network Architecture
# Define layers for image analysis
input_image = tf.keras.Input(shape=(image_shape)) # Define shape of input neurons in the network
x = tf.keras.layers.Conv2D(8, (8,8), activation='relu')(input_image) # Convolution layer with 8 2x2 filters and ReLU activation function
x = tf.keras.layers.MaxPooling2D(pool_size=(4,4))(x) # MaxPooling layer downsampling 4x4 to 1x1
x = tf.keras.layers.Dropout(0.12)(x) # Drop 12% of neurons randomly for robustness
x = tf.keras.layers.Conv2D(16, (8,8), activation='relu')(x) # Convolution layer with 16 2x2 filters and ReLU activation function
x = tf.keras.layers.MaxPooling2D(pool_size=(3,3))(x) # MaxPooling layer downsampling 3x3 to 1x1
x = tf.keras.layers.Dropout(0.12)(x) # Drop 12% of neurons randomly for robustness
x = tf.keras.layers.Conv2D(32, (2,2), activation='relu')(x) # Convolution layer with 16 2x2 filters and ReLU activation function
x = tf.keras.layers.Dropout(0.12)(x) # Drop 12% of neurons randomly for robustness
image_layers = tf.keras.layers.Flatten()(x) # Reshape input neurons to a single long vector
# Define layer for turn information
#input_turn = tf.keras.layers.Input(shape=(1)) # Define shape of input neurons in the network
# now concatenate the image and the turn layer
#base_layers = tf.keras.layers.concatenate([image_layers, input_turn])

# %%
# Head for regression
#x1 = tf.keras.layers.Dense(80, activation='relu')(base_layers) # Fully connected layer with 8 neurons and ReLU activation function
x1 = tf.keras.layers.Dense(80, activation='relu')(image_layers) # Fully connected layer with 8 neurons and ReLU activation function
x1 = tf.keras.layers.Dropout(0.12)(x1) # Drop 20% of neurons randomly for robustness
output_reg = tf.keras.layers.Dense(1, activation=None, name='reg_out')(x1) # output neuron for steering angle regression

# %%
# Head for classification
#x2 = Dense(4, activation='relu')(base_layers) # Fully connected layer with 4 neurons and ReLU activation function
#x2 = Dropout(0.2)(x2) # Drop 20% of neurons randomly for robustness
#output_class = Dense(1, activation='sigmoid', name='class_out')(x2) # output neuron for lane-in-view classification


# %%
#model = tf.keras.models.Model(inputs= [input_image, input_turn], outputs= output_reg)
model = tf.keras.models.Model(inputs= input_image, outputs= output_reg)
model.summary()
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


# %%
model.compile(
    loss={'reg_out': 'mean_squared_error'}, #, 'class_out': 'binary_crossentropy'},  # Define Error functions for outputs
    optimizer='Adam', # Gradient descent variation with dynamic learning rate
    loss_weights={'reg_out': 1}#, 'class_out': 5}  # Weighting of both loss functions E_tot = 1*E_regr + 5*E_class
)

# %%
# Network Training
history = model.fit(X, #[X, X_turn], # model input
          {'reg_out': YR}, #, 'class_out': YC}, # model outputs
          batch_size=5, #32, # number of forward passes before adjusting the weights using back propagation          
          epochs=6, #80, # how many times do we train on the entire dataset
          verbose=1,
          validation_split=0.4, # automatically create a 85%/15% test/validation data split and use for validation
          shuffle=False, # shuffle the training data order
         )

# %% Plot error history during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning Curves')
plt.ylabel('Error')
plt.xlabel('epochs')
plt.legend(['training loss', 'validation loss'])

# %% Save the model
model.save(model_name)

#model.save('/home/tom/jetson_sftp/ai_stuff/ai_jetson_nano_car/CNN_follow_tf2.h5')
# %% convert model to TRT
print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16,
                                                               max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir = model_name,
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir=model_name_trt)
print('Done Converting to TF-TRT FP16')
