from osgeo import gdal, osr, ogr
import numpy as np
import glob
Target_path = 'Data/Tar_zeros.tif'
Green_tide_set = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample'
save_path = 'Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment'
def GetExtent(geotrans,xsize,ysize):
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    ds=None
    return min_x,max_y,max_x,min_y
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
Green_tide_files = sorted(glob.glob(Green_tide_set + '\*.tif'))


for i in range(len(Green_tide_files)):
    Tar_data, Tar_proj, Tar_geotrans, Tar_row, Tar_col = read_tif(Target_path)
    Tar_min_x, Tar_max_y, Tar_max_x, Tar_min_y = GetExtent(Tar_geotrans, Tar_col, Tar_row)
    lon_lt = Tar_geotrans[0]
    lat_lt = Tar_geotrans[3]
    lon_rb = Tar_geotrans[0] + Tar_geotrans[1] * Tar_col
    lat_rb = Tar_geotrans[3] + Tar_geotrans[5] * Tar_row
    lat = np.linspace(lat_lt, lat_rb, Tar_row)
    lon = np.linspace(lon_lt, lon_rb, Tar_col)

    file_name = Green_tide_files[i].split('\\')[-1]
    print(file_name)
    Need_Fill_data, Need_Fill_proj, Need_Fill_geotrans, Need_Fill_row, Need_Fill_col = read_tif(Green_tide_files[i])
    print('Need_Fill_geotrans',Need_Fill_geotrans)
    min_x, max_y, max_x, min_y = GetExtent(Need_Fill_geotrans,Need_Fill_col,Need_Fill_row)
    lon_diff = abs(lon - min_x)
    lat_diff = abs(lat - max_y)
    lon_min = np.min(lon_diff)
    lat_min = np.min(lat_diff)
    index_row = np.where(lat_diff == lat_min)
    index_col = np.where(lon_diff == lon_min)
    row_start = index_row[0]
    print('row_start', row_start)
    col_start = index_col[0]
    print('col_start', col_start)
    print(row_start,Need_Fill_data.shape[0],col_start,Need_Fill_data.shape[1])
    print('Tar_row', Tar_row)
    print('Need_Fill_row', Need_Fill_row)
    Start_End_Diff_Row = min(Tar_row - row_start[0], Need_Fill_row)
    Start_End_Diff_Col = min(Tar_col - col_start[0], Need_Fill_col)
    print(Start_End_Diff_Row,Start_End_Diff_Col)
    Tar_data[row_start[0]:row_start[0]+Start_End_Diff_Row, col_start[0]:col_start[0]+Start_End_Diff_Col] = Need_Fill_data[:Start_End_Diff_Row,:Start_End_Diff_Col]

    write_tif(save_path + '\\' + file_name, Tar_data, Tar_geotrans, Tar_proj)
    del Need_Fill_data,Tar_data,lon_diff,lat_diff

