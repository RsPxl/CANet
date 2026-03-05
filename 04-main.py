import glob
from model import *
import random
from read_gdal import GRID
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import InteractiveSession
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.device('/gpu:0')
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = InteractiveSession(config=config)
batch_size = 4

train_img = 'Data/DataSet_BandCombine_Train/Img_SampleSet_DelBackground'
train_lab = 'Data/DataSet_BandCombine_Train/Lab_SampleSet_DelBackground'
img_list = sorted(glob.glob(train_img+'\*.tif'))
lab_list = sorted(glob.glob(train_lab+'\*.tif'))
img_length = len(glob.glob(train_img+'\*.tif'))


def generate_train_batch(batch_size):
    '''Yield a generator of training data from filename on given list of cols split for train/test'''
    while True:
        x_batch = []
        y_batch = []
        for b in range(batch_size):
            i = random.randint(0,img_length-1)
            x = GRID().read_tif(img_list[i])
            y = GRID().read_tif(lab_list[i])
            x_batch.append(x)
            y_batch.append(y)
            del x,y
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        # yield x_batch, y_batch[:,:,:,np.newaxis]
        yield x_batch, y_batch
        del x_batch, y_batch


model = unet_CA_GHM()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(generate_train_batch(batch_size),steps_per_epoch=300,epochs=1000)#,callbacks=[model_checkpoint]
model.save('logs/CANet.h5')
