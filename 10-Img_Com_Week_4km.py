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

def weekly_combine(year, month, path_list,num,Output_folder):
    if len(path_list) == 0:
        print('List', year,'_',month,'_',num, 'is empty')
    else:
        np_zeros = np.zeros((2488,2370,1),np.float16)
        for day_num in range(len(path_list)):
            print(path_list[day_num])
            Tar_data, Tar_proj, Tar_geotrans, Tar_row, Tar_col = read_tif(path_list[day_num])
            Tar_data[Tar_data == 4] = np.nan
            Tar_data[Tar_data == 2] = np.nan
            Tar_data[Tar_data == 3] = 0
            np_zeros = np.concatenate((np_zeros,Tar_data[:,:,np.newaxis]),axis=2)
        np_zeros = np_zeros[:,:,1:]
        row = int(np_zeros.shape[0] / 16)
        col = int(np_zeros.shape[1] / 16)
        data = np.zeros((row, col), np.float16)
        for r in range(row):
            for c in range(col):
                data[r, c] = np.nanmean(np_zeros[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16, :])
        geotrans_save = (Tar_geotrans[0], 0.04, 0, Tar_geotrans[3], 0, -0.04)
        write_tif(Output_folder+'\\'+str(year)+'_'+str(month)+'_'+str(num)+'.tif', data, geotrans_save, Tar_proj)
        del Tar_data,data
def main():
    CANet_path = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment'
    Output_folder = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment_Composition'
    month_list = [5, 6, 7, 8]   #months for macroalgae bloom
    for i in range(2013,2015,1):
        for j in month_list:
            Year_month_list = sorted(glob.glob(CANet_path + '\\%d' % i + '*_'+'%.2d' % j +'*_*.tif'))
            week_1, week_2, week_3, week_4 = [], [], [], []
            for w in range(len(Year_month_list)):
                day = int(Year_month_list[w].split('\\')[-1].split('_')[1][2:])
                if  day < 8:
                    week_1.append(Year_month_list[w])
                elif day >= 8 and day < 15:
                    week_2.append(Year_month_list[w])
                elif day >= 15 and day < 22:
                    week_3.append(Year_month_list[w])
                else:
                    week_4.append(Year_month_list[w])
            weekly_combine (i,j,week_1,1,Output_folder)
            weekly_combine (i,j,week_2,2,Output_folder)
            weekly_combine (i,j,week_3,3,Output_folder)
            weekly_combine (i,j,week_4,4,Output_folder)
if __name__ == '__main__':
    main()

