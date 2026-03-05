from model import *
import shutil,sys
from read_gdal import GRID
from keras.models import Sequential,load_model
from osgeo import gdal_array as ga
from osgeo import gdal
import glob
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
IMAGE_SIZE = 128

model = load_model('logs/CANet.h5', custom_objects={'ca_block': ca_block, 'se_block': se_block, 'tf': tf,
                                                             'focal_loss': focal_loss})  # ,'focal_loss_fixed': focal_loss_fixed
image_path = 'Data/DataSet_BandCombine_Test'
predict_path = 'Data/DataSet_BandCombine_Test_Result'

def split_to_net(ori_image,size,sample_image_path):

    print('Start...')
    path = 'sample_image_for_predict/'
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
    print('ori_image.shape',ori_image.shape)
    h_step = ori_image.shape[0] // (size-50)+1
    if (h_step-2) * size + size - 50 * (h_step-2) > ori_image.shape[0]:
        h_step = ori_image.shape[0] // (size-50)
    w_step = ori_image.shape[1] // (size-50)+1
    if (w_step-2) * size + size - 50 * (w_step-2) > ori_image.shape[1]:
        w_step = ori_image.shape[1] // (size - 50)
    count = 1
    for h in range(h_step):
        for w in range(w_step):
            if h == 0:
                if w == 0:
                    image_sample = ori_image[(h * size):(h * size + size),
                                   (w * size):(w * size + size), :]
                elif w == w_step - 1:
                    image_sample = ori_image[(h * size):(h * size + size),
                                   (ori_image.shape[1] - size): ori_image.shape[1], :]
                else:
                    image_sample = ori_image[(h * size):(h * size + size),
                                   (w * size - 50 * w):(w * size + size - 50 * w), :]
            elif h == h_step - 1:
                if w == 0:
                    image_sample = ori_image[ori_image.shape[0] - size:ori_image.shape[0],
                                   (w * size):(w * size + size), :]
                elif w == w_step - 1:
                    image_sample = ori_image[ori_image.shape[0] - size:ori_image.shape[0],
                                   (ori_image.shape[1] - size): ori_image.shape[1], :]
                else:
                    image_sample = ori_image[ori_image.shape[0] - size:ori_image.shape[0],
                                   (w * size - 50 * w):(w * size + size - 50 * w), :]
            else:
                if w == 0:
                    image_sample = ori_image[(h * size - 50 * h):(h * size + size - 50 * h),
                                   (w * size):(w * size + size), :]
                elif w == w_step - 1:
                    image_sample = ori_image[(h * size - 50 * h):(h * size + size - 50 * h),
                                   (ori_image.shape[1] - size):ori_image.shape[1], :]
                else:
                    image_sample = ori_image[(h * size - 50 * h):(h * size + size - 50 * h),
                                   (w * size - 50 * w):(w * size + size - 50 * w), :]
            image_path = path + str(count) + '.tif'

            GRID().write_tif(image_path, image_sample)
            sample_image_path.append(image_path)
            count += 1
            print('count',count)
    del ori_image,image_sample
    return h_step, w_step
def combin_image(ori_image_path,ori_image, h_step, w_step, predict_path,  sample_image_path):
    tmp = np.ones([ori_image.shape[0], ori_image.shape[1]],np.int8)
    for h in range(h_step):
        for w in range(w_step):
            if h == 0:
                if w == 0:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[h * IMAGE_SIZE:(h + 1) * IMAGE_SIZE - 25, w * IMAGE_SIZE:(w + 1) * IMAGE_SIZE - 25] = img_now[0:IMAGE_SIZE - 25, 0:IMAGE_SIZE - 25]
                    del img_now
                elif w == w_step-1:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[h * IMAGE_SIZE:(h + 1) * IMAGE_SIZE - 25, 25-IMAGE_SIZE:ori_image.shape[1]] = \
                        img_now[0:IMAGE_SIZE - 25, 25-IMAGE_SIZE:IMAGE_SIZE]
                    del img_now
                else:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[h * IMAGE_SIZE:(h + 1) * IMAGE_SIZE - 25,
                    (IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * (w - 1):(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * w] = img_now[ 0:IMAGE_SIZE - 25, 25:IMAGE_SIZE - 25]
                    del img_now
            elif h == h_step-1:
                if w == 0:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[25-IMAGE_SIZE:ori_image.shape[0], w * IMAGE_SIZE:(w + 1) * IMAGE_SIZE - 25] = img_now[25-IMAGE_SIZE: IMAGE_SIZE, 0:IMAGE_SIZE - 25]
                    del img_now
                elif w == w_step - 1:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[25-IMAGE_SIZE:ori_image.shape[0],
                    25-IMAGE_SIZE:ori_image.shape[1]] = \
                        img_now[25-IMAGE_SIZE:ori_image.shape[0],
                        25-IMAGE_SIZE:IMAGE_SIZE]
                    del img_now
                else:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[25-IMAGE_SIZE:ori_image.shape[0],
                    (IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * (w - 1):(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * w] = \
                        img_now[25-IMAGE_SIZE : IMAGE_SIZE, 25:IMAGE_SIZE - 25]
                    del img_now
            else:
                if w == 0:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * (h - 1):(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * h,
                    w * IMAGE_SIZE:(w + 1) * IMAGE_SIZE - 25] = img_now[25:IMAGE_SIZE - 25, 0:IMAGE_SIZE - 25]
                    del img_now
                elif w == w_step - 1:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * (h - 1):(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * h,
                    25-IMAGE_SIZE:ori_image.shape[1]] = \
                        img_now[25:IMAGE_SIZE - 25, 25-IMAGE_SIZE:IMAGE_SIZE]
                    del img_now
                else:
                    img_now = GRID().read_tif(sample_image_path[h * w_step + w])
                    tmp[(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * (h - 1):(IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * h,
                    (IMAGE_SIZE - 25) + (IMAGE_SIZE - 50) * (w - 1):(IMAGE_SIZE - 25) + (
                            IMAGE_SIZE - 50) * w] = img_now[25:IMAGE_SIZE - 25, 25:IMAGE_SIZE - 25]
                    del img_now
    Nan_position = ori_image[:, :, 0] + ori_image[:, :, 1]
    tmp[Nan_position == 0] = 4
    tar = gdal.Open(ori_image_path)
    ga.SaveArray(tmp, predict_path, format="GTiff", prototype=tar)
    del tar, tmp, Nan_position
img_list = sorted(glob.glob(image_path+'\\*.tif'))

for i in range(len(img_list)):
    name = img_list[i].split('\\')[-1]
    sample_image_path = []
    ori_image = GRID().read_tif(img_list[i])
    h_step, w_step = split_to_net(ori_image, IMAGE_SIZE, sample_image_path)

    for path in sample_image_path:
        x = np.expand_dims(GRID().read_tif(path),axis=0)
        pred = model.predict(x)
        class_indices = np.argmax(pred, axis=3)
        GRID().write_tif(path,np.squeeze(class_indices))
        del x, pred, class_indices
    sys.stdout.flush()
    combin_image(img_list[i],ori_image,h_step,w_step,predict_path+'\\'+name,sample_image_path)
    print('end')
    del ori_image

