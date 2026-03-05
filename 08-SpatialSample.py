# -*- coding: utf-8 -*-
import numpy as np
from osgeo import gdal, osr, ogr
import glob
from scipy.interpolate import griddata


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

def Tif_mapping(Img_data, LatLon_Before, Img_proj, Tar_Lat, Tar_Lon, name, Output_folder):


    Tar_Lons, Tar_Lats = np.meshgrid(Tar_Lon, Tar_Lat)  #  (249,237)
    Tar_LonMin, Tar_LatMax, Tar_LonMax, Tar_LatMin = [Tar_Lon.min(), Tar_Lat.max(), Tar_Lon.max(), Tar_Lat.min()]



    im_data_need_insert = Img_data.reshape(-1, 1)  #(249*237,1)
    Data_griddata_After = griddata(LatLon_Before, im_data_need_insert,(Tar_Lats, Tar_Lons), method='nearest').squeeze()
    driver = gdal.GetDriverByName('GTiff')
    out_tif = driver.Create(Output_folder+'\\'+ name, Data_griddata_After.shape[1], Data_griddata_After.shape[0], 1, gdal.GDT_Float32)
    geotransform = (Tar_LonMin, 0.0025, 0, Tar_LatMax, 0, -0.0025)
    out_tif.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_tif.SetProjection(srs.ExportToWkt())
    out_tif.GetRasterBand(1).WriteArray(Data_griddata_After)
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
    img_path = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing'
    img_list = sorted(glob.glob(img_path + '\*.tif'))
    Output_folder = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample'
    for i in range(len(img_list)):
        name = img_list[i].split('\\')[-1]

        Img_data, Img_proj, Img_geotrans, Img_row, Img_col = read_tif(img_list[i])
        Img_lon_lt = Img_geotrans[0]
        Img_lat_lt = Img_geotrans[3]
        Img_lon_rb = Img_geotrans[0] + Img_geotrans[1] * Img_col
        Img_lat_rb = Img_geotrans[3] + Img_geotrans[5] * Img_row
        Lat = np.linspace(Img_lat_lt, Img_lat_rb, Img_row)
        Lon = np.linspace(Img_lon_lt, Img_lon_rb, Img_col)
        Lons, Lats = np.meshgrid(Lon, Lat)
        LatLon_Before = np.hstack((Lats.reshape(-1, 1), Lons.reshape(-1, 1)))  # (249*237,2)


        Tar_col = int((Img_lon_rb - Img_lon_lt) / 0.0025)
        Tar_row = int((Img_lat_lt - Img_lat_rb) / 0.0025)

        Tar_lon_rb = Img_geotrans[0] + 0.0025 * Tar_col
        Tar_lat_rb = Img_geotrans[3] + (-0.0025) * Tar_row
        Tar_Lat = np.linspace(Img_geotrans[3], Tar_lat_rb, Tar_row)
        Tar_Lon = np.linspace(Img_geotrans[0], Tar_lon_rb, Tar_col)


        Tif_mapping(Img_data,LatLon_Before,Img_proj, Tar_Lat, Tar_Lon, name,Output_folder)

        print ('-----'+name+'!')
        del Img_data,LatLon_Before,Lons, Lats
if __name__ == '__main__':
    main()

