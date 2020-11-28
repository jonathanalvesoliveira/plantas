import os
import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from tensorflow.keras.applications import DenseNet121
from sklearn.model_selection import train_test_split
import tensorflow.keras.layers as L

import tensorflow.compat.v1 as tfc
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto ()
config.gpu_options.allow_growth = True
session = InteractiveSession (config = config)

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
    

np.random.seed(0)
tf.random.set_seed(0)

tqdm.pandas()

EPOCHS = 40
#BATCH_SIZE = 10
IMAGE_SIZE = 500#512
AUTO = tf.data.experimental.AUTOTUNE
IMAGE_PATH = "./imagens/"
TEST_PATH = "./test.csv"
TRAIN_PATH = "./train.csv"
SUB_PATH = "./sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)

def format_path(st):
    return IMAGE_PATH+st+'.jpg'

train_paths = train_data.image_id.apply(format_path).values

train_labels = np.float32(train_data.loc[:, 'healthy':'scab'].values)
train_paths, valid_paths, train_labels, valid_labels =\
train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)
test_paths = test_data.image_id.apply(format_path).values

def decode_image(filename, label=None, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label
    
strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = 16 * strategy.num_replicas_in_sync#16
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)

def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn
lrfn = build_lrfn()

model = tf.keras.Sequential()
model.add(tf.keras.applications.ResNet50V2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', include_top=False))
model.add(L.GlobalAveragePooling2D())
model.add(L.Dense(train_labels.shape[1], activation='softmax'))

model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
model.summary()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

net_name = 'resnet_'
checkpoint_path = "./training_"+net_name+"_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule, cp_callback],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)

print(history)

# Restore the weights
#model.load_weights(checkpoint_path)

# Evaluate the model
loss, acc = model.evaluate(valid_dataset)
print("VALID DATASET ACCURACY: {:5.2f}%".format(100 * acc))

probs_efnns = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_efnns
sub.to_csv('submission_'+net_name+'.csv', index=False)
sub.head()
