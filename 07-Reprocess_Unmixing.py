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

def doy2date(year, doy):
    month_leapyear = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_notleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        for i in range(0, 12):
            if doy > month_leapyear[i]:
                doy -= month_leapyear[i]
                continue
            if doy <= month_leapyear[i]:
                month = i + 1
                day = doy
                break
    else:
        for i in range(0, 12):
            if doy > month_notleap[i]:
                doy -= month_notleap[i]
                continue
            if doy <= month_notleap[i]:
                month = i + 1
                day = doy
                break
    return month, day
def main():
    FAI_path = 'Data/DataSet_BandCombine_Test'
    CANet_path = 'Data/DataSet_BandCombine_Test_Result_Filter'
    data_with_Ulva_list = sorted(glob.glob(CANet_path + '\*.tif'))
    Output_folder = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing'
    for i in range(len(data_with_Ulva_list)):
        name = data_with_Ulva_list[i].split('\\')[-1]
        print(name)
        CANet_data, CANet_proj, CANet_geotrans, CANet_row, CANet_col = read_tif(data_with_Ulva_list[i])
        FAI_Data_path = FAI_path+'\\'+name
        FAI_Ddata, FAI_Dproj, FAI_Dgeotrans, FAI_Drow, FAI_Dcol = read_tif(FAI_Data_path)
        FAI = FAI_Ddata[:,:,-1]

        FAI[np.where(FAI > 0.3)] = 0.3
        FAI[np.where(FAI < -0.0015)] = -0.0015

        FAI_Unmix = (FAI + 0.002) / (0.192 + 0.002)
        FAI_Unmix[np.where(FAI_Unmix > 1)] = 1
        FAI_Unmix[np.where(FAI_Unmix < 0)] = 0
        FAI_UnmixAndClasses = np.where(CANet_data==1,FAI_Unmix,CANet_data)


        write_tif(Output_folder + '\\' + name, FAI_UnmixAndClasses, CANet_geotrans, CANet_proj)
if __name__ == '__main__':
    main()

