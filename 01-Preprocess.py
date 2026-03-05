# -*- coding: utf-8 -*-

import numpy as np
from osgeo import gdal, osr, ogr
import glob


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



def main():
    img_path = 'Data/DataSet'
    img_list = sorted(glob.glob(img_path + '\*.tif'))
    Output_Img = 'Data/DataSet_BandCombine'
    for i in range(len(img_list)):
        img_data, img_proj, img_geotrans, img_height, img_width = read_tif(img_list[i])
        name_save = img_list[i].split('\\')[-1]
        print('name_save',name_save)
        #FAI
        Rred = img_data[:, :, 0]
        Rnir = img_data[:, :, 1]
        Rswir = img_data[:, :, 4]
        FAI = Rnir - (Rred + (Rswir - Rred) * (859 - 645) / (1240 - 645))
        #delete 1640
        Img_Save = np.zeros((img_data.shape[0],img_data.shape[1],7))
        Img_Save[:,:,:5] = img_data[:,:,:5]
        Img_Save[:,:,5] = img_data[:,:,6]
        Img_Save[:, :, 6] = FAI
        write_tif(Output_Img+'\\'+name_save, Img_Save, img_geotrans, img_proj)
if __name__ == '__main__':
    main()

