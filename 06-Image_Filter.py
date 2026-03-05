# -*- coding: utf-8 -*-

import glob
from osgeo import gdal, osr, ogr
import cv2
import numpy as np
def read_tif(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    Bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    if len(im_data.shape) == 2:
        return im_data, im_proj, im_geotrans, im_height, im_width
    else:
        im_data = np.zeros([im_height, im_width, Bands])
        for j in range(Bands):
            raster_band = dataset.GetRasterBand(j + 1)
            im_data[:, :, j] = raster_band.ReadAsArray()
        return im_data, im_proj, im_geotrans, im_height, im_width
    del dataset
def write_tif(FileName,data,geotrans,proj):
    if len(data.shape) == 3:
        height, width, bands = data.shape
    else:
        (height, width), bands, = data.shape, 1
    driver = gdal.GetDriverByName('GTiff')
    out_tif = driver.Create(FileName, width, height, bands, gdal.GDT_Float32)
    geotransform = geotrans
    out_tif.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_tif.SetProjection(proj)

    if bands == 1:
        out_tif.GetRasterBand(1).WriteArray(data)
    else:
        for m in range(bands):
            out_tif.GetRasterBand(m + 1).WriteArray(data[:, :, m])
    out_tif.FlushCache()
    del out_tif
path = 'Data/DataSet_BandCombine_Test_Result'
save_path = 'Data/DataSet_BandCombine_Test_Result_Filter'
path_list = sorted(glob.glob(path+'\*.tif'))
for i in range(len(path_list)):
    name = path_list[i].split('\\')[-1]
    print(name)
    img_data, img_proj, img_geotrans, img_height, img_width = read_tif(path_list[i])
    print(img_data.dtype)
    img_data = img_data.astype(np.int16)
    print(img_data.dtype)
    img_data[img_data == 2] = 1000
    kernel = np.array([[1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49], [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49], [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49], [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49], [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],[1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],[1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49] ])
    img = cv2.filter2D(img_data, -1, kernel)
    img_data [img>400] =1000
    img_data [img_data==1000] = 2
    write_tif(save_path+'\\'+name, img_data ,img_geotrans, img_proj)
    del img_data,img