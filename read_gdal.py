# -*- coding: utf-8 -*-

import numpy as np
from osgeo import gdal
global im_geotrans, im_proj
class GRID:

    def read_tif(self,filename):
        global im_geotrans, im_proj
        dataset=gdal.Open(filename)
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        Bands = dataset.RasterCount
        im_geotrans = dataset.GetGeoTransform()
        im_proj = dataset.GetProjection()
        im_data = dataset.ReadAsArray(0,0,im_width,im_height)
        if len(im_data.shape) == 2:

            return im_data
        else:
            im_data = np.zeros([im_height, im_width, Bands],im_data.dtype)
            for j in range(Bands):
                raster_band = dataset.GetRasterBand(j + 1)
                im_data[:, :, j] = raster_band.ReadAsArray()


        del dataset 
        
        return im_data


    def write_tif(self,filename,im_data):
        global im_geotrans, im_proj

        #gdal.GDT_Byte, 
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64


        datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_height, im_width,im_bands = im_data.shape
        else:
            (im_height, im_width),im_bands, = im_data.shape,1

        
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[:,:,i])

        del dataset
