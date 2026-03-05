
# -*- coding: utf-8 -*-
import numpy as np
from osgeo import gdal, osr, ogr
import glob
import re


def fuzzy_search(pattern, sequence):
    regex = re.compile(pattern)
    return [item for item in sequence if regex.search(item)]
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
def write_tif(FileName, data, geotrans, proj):
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
    CANet_path = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment'
    Output_folder = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment_Composition'
    for i in range(2016,2017,1):
        File_name_list = sorted(glob.glob(CANet_path+'\\%d'%i+'*.tif'))
        np_zeros = np.zeros((2488,2370,1),np.float16)
        print('len(File_name_list)',len(File_name_list))
        for j in range(len(File_name_list)):
            print(File_name_list[j].split('\\')[-1])
            CANet_data, CANet_proj, CANet_geotrans, CANet_row, CANet_col = read_tif(File_name_list[j])
            CANet_data[CANet_data == 4] = np.nan
            CANet_data[CANet_data == 2] = np.nan
            CANet_data[CANet_data == 3] = 0
            np_zeros = np.concatenate((np_zeros,CANet_data[:,:,np.newaxis]),axis=2)
        np_zeros = np_zeros[:,:,1:]
        row = int(np_zeros.shape[0] / 16)
        col = int(np_zeros.shape[1] / 16)
        data = np.zeros((row, col), np.float16)
        for r in range(row):
            for c in range(col):
                data[r, c] = np.nanmean(np_zeros[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16, :])
        geotrans_save = (CANet_geotrans[0], 0.04, 0, CANet_geotrans[3], 0, -0.04)
        write_tif(Output_folder + '\\' + '%d'%i+'.tif', data, geotrans_save, CANet_proj)
        del CANet_data, data

if __name__ == '__main__':
    main()

