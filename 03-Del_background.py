import numpy as np
import os
from read_gdal import GRID
import glob

os.environ["CUDA_VISIBLE_DEVICES"]="0"
labWithbackground = 'Data/DataSet_BandCombine_Train/Lab_SampleSet'
imgWithbackground = 'Data/DataSet_BandCombine_Train/Img_SampleSet'
lab_path = 'Data/DataSet_BandCombine_Train/Lab_SampleSet_DelBackground'
img_path = 'Data/DataSet_BandCombine_Train/Img_SampleSet_DelBackground'

lab_list = sorted(glob.glob(labWithbackground+'\*.tif'))
img_list = sorted(glob.glob(imgWithbackground+'\*.tif'))
for path_num in range(len(lab_list)):
    name = lab_list[path_num].split('\\')[-1]
    lab = GRID().read_tif(lab_list[path_num])
    if np.sum(lab[:,:,1]) > 300:
        GRID().write_tif(lab_path + '\\' + name, lab)
        img = GRID().read_tif(img_list[path_num])
        GRID().write_tif(img_path + '\\' + name, img)
