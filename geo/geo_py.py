from netCDF4 import Dataset,MFDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap,cm
from osgeo import gdal,ogr,osr
import numpy as np
from glob import glob
import json
import IPython
import shapefile
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib as mpl
from shapely.geometry import Polygon
import geopandas as gpd
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import sys

font_path = os.path.join('geo','SimHei.ttf')

def read_tif(tif_path):
    ds = gdal.Open(tif_path)
    row = ds.RasterXSize
    col = ds.RasterYSize
    band = ds.RasterCount

    for i in range(band):
        data = ds.GetRasterBand(i+1).ReadAsArray()

        data = np.expand_dims(data , 2)
        if i == 0:
            allarrays = data
        else:
            allarrays = np.concatenate((allarrays, data), axis=2)
    return {'data':allarrays,'transform':ds.GetGeoTransform(),'projection':ds.GetProjection(),'bands':band,'width':row,'height':col}
    # 左上角点坐标 GeoTransform[0],GeoTransform[3] Transform[1] is the pixel width, and Transform[5] is the pixel height

def write_tif(fn_out, im_data, transform,proj=None):
    '''
    功能:
    ----------
    将矩阵按某种投影写入tif，需指定仿射变换矩阵，可选渲染为rgba
    
    参数:
    ----------
    fn_out:str
        输出tif图的绝对文件路径
    im_data: np.array
        tif图对应的矩阵
    transform: list/tuple 
        gdal-like仿射变换矩阵，若im_data矩阵起始点为左上角且投影为4326，则为
            (lon_x.min(), delta_x, 0, 
             lat_y.max(), 0, delta_y)
    proj: str（wkt格式）
        投影，默认投影坐标为4326，可用osr包将epsg转化为wkt格式，如
            srs = osr.SpatialReference()# establish encoding
            srs.ImportFromEPSG(4326)    # WGS84 lat/lon
            proj = srs.ExportToWkt()    # create wkt fromat of proj

    '''
    # 设置投影，proj为wkt format
    if proj is None:
        proj = 'GEOGCS["WGS 84",\
                     DATUM["WGS_1984",\
                             SPHEROID["WGS 84",6378137,298.257223563, \
                                    AUTHORITY["EPSG","7030"]], \
                             AUTHORITY["EPSG","6326"]], \
                     PRIMEM["Greenwich",0, \
                            AUTHORITY["EPSG","8901"]], \
                     UNIT["degree",0.0174532925199433, \
                            AUTHORITY["EPSG","9122"]],\
                     AUTHORITY["EPSG","4326"]]'
    # 设置数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 将(通道数、高、宽)顺序调整为(高、宽、通道数)
    # print('shape of im data:', im_data.shape)
    im_bands = min(im_data.shape)
    im_shape = list(im_data.shape)
    im_shape.remove(im_bands)
    im_height, im_width = im_shape
    band_idx = im_data.shape.index(im_bands)
    # 找出波段是在第几个

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fn_out, im_width, im_height, im_bands, datatype)

    # if dataset is not None:
    dataset.SetGeoTransform(transform)  # 写入仿射变换参数
    dataset.SetProjection(proj)  # 写入投影

    if im_bands == 1:
        # print(im_data[:, 0,:].shape)
        if band_idx == 0:
            dataset.GetRasterBand(1).WriteArray(im_data[0, :, :])
        elif band_idx == 1:
            dataset.GetRasterBand(1).WriteArray(im_data[:, 0, :])
        elif band_idx == 2:
            dataset.GetRasterBand(1).WriteArray(im_data[:, :, 0])

    else:
        for i in range(im_bands):
            if band_idx == 0:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i, :, :])
            elif band_idx == 1:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, i, :])
            elif band_idx == 2:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])

    dataset.FlushCache()
    del dataset
    driver = None

def NC_to_tifs(data, Output_folder):
    nc_data_obj = nc.Dataset(data)
    Lon = nc_data_obj.variables['longitude'][:]
    Lat = nc_data_obj.variables['latitude'][:]
    ndvi_arr = np.asarray(nc_data_obj.variables['blh'])  # 将ndvi数据读取为数组
    ndvi_arr_float = ndvi_arr.astype(float) / 10000  # 将int类型改为float类型,并化为-1 - 1之间

    # 影像的左上角和右下角坐标
    LonMin, LatMax, LonMax, LatMin = [Lon.min(), Lat.max(), Lon.max(), Lat.min()]

    # 分辨率计算
    N_Lat = len(Lat)
    N_Lon = len(Lon)
    Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)

    for i in range(len(ndvi_arr[:])):
        driver = gdal.GetDriverByName('GTif')
        out_tif_name = Output_folder + '/' + data.split('/')[-1].split('.')[0] + '_' + str(i + 1) + '.tif'
        out_tif = driver.Create(out_tif_name, N_Lon, N_Lat, 1, gdal.GDT_Float32)

        # 设置影像的显示范围
        # -Lat_Res一定要是-的
        geotransform = (LonMin, Lon_Res, 0, LatMax, 0, -Lat_Res)
        out_tif.SetGeoTransform(geotransform)

        # 获取地理坐标系统信息，用于选取需要的地理坐标系统
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # 定义输出的坐标系为"WGS 84"，AUTHORITY["EPSG","4326"]
        out_tif.SetProjection(srs.ExportToWkt())  # 给新建图层赋予投影信息

        # 数据写出
        print(ndvi_arr[i][0])
        out_tif.GetRasterBand(1).WriteArray(ndvi_arr[i][0])  # 将数据写入内存，此时没有写入硬盘
        out_tif.FlushCache()  # 将数据写入硬盘
        out_tif = None  # 注意必须关闭tif文件


def avg_tif(tifs):
    datas = [read_tif(tif)['data'] for tif in tifs]
    for data in datas:
        data[np.isnan(data)] = 0
    if len(set([data.shape for data in datas])) != 1:
        raise Exception('tifs shapes are not same!')  
    sum_data = datas[0]
    for data in datas[1:]:
        sum_data += data
    shape = datas[0].shape
    counts = np.zeros(shape)
    # remove 0 value
    for data in datas:
        zero_indexs = np.nonzero(data)
        for i,j in zip(zero_indexs[0],zero_indexs[1]):
            counts[i][j] += 1
    avg_data = np.divide(sum_data,counts)
    avg_data[np.isnan(avg_data)] = 0
    avg_data[np.isinf(avg_data)] = 0
    ds = gdal.Open(tifs[0])
    return {'data':avg_data,'transform':ds.GetGeoTransform(),'projection':ds.GetProjection()}

def tif_division(divisor,Dividend):
    d1 = read_tif(divisor)
    d2 = read_tif(Dividend)
    data = np.divide(d1['data'],d2['data'])
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    return {'data':data,'transform':d1['transform'],'projection':d1['projection']}

def SRS_transform(src,dst,dst_srs='EPSG:4326'):
    ds = gdal.Warp(srcDSOrSrcDSTab=src,
            destNameOrDestDS=dst,
            format='GTif',
            dstNodata=0,
            width=1280,
            height=820,
            dstSRS=dst_srs)
    ds = None # 释放资源，否则不会成功生成tif

def tif_Resample(src,dst,x,y):
    ds = gdal.Warp(srcDSOrSrcDSTab=src,
        destNameOrDestDS=dst,
        format='GTif',
        xRes=x, yRes=y,
        dstNodata=0)
    ds = None # 释放资源，否则不会成功生成tif

def tif_merge(output_tif,tifs):
    os.system('gdal_merge.py -init 0 -o {} {}'.format(output_tif,' '.join(tifs)))

def tif_to_shp(tif,shp):
    type_mapping = {gdal.GDT_Byte: ogr.OFTInteger,
                    gdal.GDT_UInt16: ogr.OFTInteger,
                    gdal.GDT_Int16: ogr.OFTInteger,
                    gdal.GDT_UInt32: ogr.OFTInteger,
                    gdal.GDT_Int32: ogr.OFTInteger,
                    gdal.GDT_Float32: ogr.OFTReal,
                    gdal.GDT_Float64: ogr.OFTReal,
                    gdal.GDT_CInt16: ogr.OFTInteger,
                    gdal.GDT_CInt32: ogr.OFTInteger,
                    gdal.GDT_CFloat32: ogr.OFTReal,
                    gdal.GDT_CFloat64: ogr.OFTReal}
    ds = gdal.Open(tif)
    pj = ds.GetProjection()
    srcband = ds.GetRasterBand(1)
    dst_layername = 'Shape'
    drv = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = drv.CreateDataSource(shp)
    spatial = osr.SpatialReference()
    spatial.ImportFromWkt(pj)
    dst_layer = dst_ds.CreateLayer(dst_layername, spatial)
    raster_field = ogr.FieldDefn('class', type_mapping[srcband.DataType])
    dst_layer.CreateField(raster_field)
    gdal.Polygonize(srcband, None, dst_layer, 0, [], callback=None)
    # gpd.GeoDataFrame(gdf, crs='EPSG:4326')

def shp_to_tif(shp, tif, column):
    input_shp = ogr.Open(shp)
    shp_layer = input_shp.GetLayer()

    # 遍历所有要素，开始读取和写入
    feature = shp_layer.GetNextFeature()
    res = []
    while feature:
        # 读取ID、cover字段值
        value = float(feature.GetFieldAsString(column))
        res.append(value)
        # 清除缓存并获取下一个要素
        feature.Destroy()
        feature = shp_layer.GetNextFeature()

    pixel_size = 0.03
    xmin, xmax, ymin, ymax = shp_layer.GetExtent()

    x_res = int((xmax - xmin) / pixel_size)
    y_res = int((ymax - ymin) / pixel_size)

    target_ds = gdal.GetDriverByName('GTiff').Create(tif, x_res, y_res, 1, gdal.GDT_Float32)

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)

    target_ds.SetProjection(outRasterSRS.ExportToWkt())
    target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))

    band = target_ds.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    ds = gdal.RasterizeLayer(target_ds, [1], shp_layer, options=["ATTRIBUTE=VAR_VALUE"])
    ds = None

def interpolate(shp,tif,column):
    opts = gdal.GridOptions(
        algorithm="invdistnn:power=3.0:smothing=200.0:radius=10:max_points=20:min_points=0:nodata=0.0",
        format="GTiff", outputType=gdal.GDT_Float32, zfield=column)
    gdal.Grid(destName=tif, srcDS=shp, options=opts)

def rasterizeLayer(inShp, outTif):
    input_shp = ogr.Open(inShp)
    shp_layer = input_shp.GetLayer()

    # 遍历所有要素，开始读取和写入
    feature = shp_layer.GetNextFeature()
    res = []
    while feature:
        # 读取ID、cover字段值
        value = float(feature.GetFieldAsString('osm_id'))
        res.append(value)
        # 清除缓存并获取下一个要素
        feature.Destroy()
        feature = shp_layer.GetNextFeature()

    pixel_size = 0.003
    xmin, xmax, ymin, ymax = shp_layer.GetExtent()

    x_res = int((xmax - xmin) / pixel_size)
    y_res = int((ymax - ymin) / pixel_size)

    target_ds = gdal.GetDriverByName('GTiff').Create(outTif, x_res, y_res, 1, gdal.GDT_Float32)

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)

    target_ds.SetProjection(outRasterSRS.ExportToWkt())
    target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))

    band = target_ds.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    ds = gdal.RasterizeLayer(target_ds, [1], shp_layer, options=["ATTRIBUTE=osm_id"])
    ds = None

def if_overlap(tif,shp):
    def read_tif(tif):
        ds = gdal.Open(tif)
        X = ds.RasterXSize
        Y = ds.RasterYSize
        band = ds.RasterCount
        data = ds.GetRasterBand(1).ReadAsArray()
        projection = ds.GetProjection()
        return {'data':data,'transform':ds.GetGeoTransform(),'projection':ds.GetProjection(),'x_size':X,'y_size':Y}
        # # 左上角点坐标 GeoTransform[0],GeoTransform[3] Transform[1] is the pixel width, and Transform[5] is the pixel height

    def SRS_transform(dst,src,dst_srs='EPSG:4326'):
        ds = gdal.Warp(srcDSOrSrcDSTab=src,
                destNameOrDestDS=dst,
                format='GTiff',
                dstNodata=0,
                width=1280,
                height=820,
                dstSRS=dst_srs)
        ds = None # 释放资源，否则不会成功生成tiff  

    SRS_transform('new.tif',tif)
    tif = read_tif('new.tif')
    x1 = tif['transform'][0]
    x2 = tif['transform'][0]+tif['transform'][1]*tif['x_size']
    y1 = tif['transform'][3]
    y2 = tif['transform'][3]+tif['transform'][5]*tif['y_size']
    tif_polygon = Polygon([[x2,y2],[x2,y1],[x1,y2],[x1,y1]])
    shp = gpd.read_file(shp)

    return shp.intersects(tif_polygon).any()


def add_watermark(from_path,to_path):
    # 有时候生成的图片拉到gis中显示是纯黑色，原因是gis软件默认选择的数值范围的问题，可以做归一化或者修改软件中的设置
    def addmark_1_band(array1,array2):
        idxs = np.where(array2 > 0)
        max = array1.max()
        for x,y in tqdm(zip(idxs[0],idxs[1])):
            try:
                array1[x,y] = max
                # 单波段无法按多波段增加透明度
            except:
                continue
        return array1

    def addmark_m_band(array1,array2):
        idxs = np.where(array2 > 0)
        max = array1.max()
        for x,y in tqdm(zip(idxs[0],idxs[1])):
            try:
                # array1[x,y] = array1[x,y] * 0.85 #random.randint(80,90) * 0.01
                array1[x, y] = 255
                # 这样既保留了原始数据，又增加了水印效果
            except:
                continue
        return array1

    def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
        draw.text(position, text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def generate_watermark(xsize,ysize):
        image = np.zeros((820,1280,4),np.uint8)
        # 先生成较小的图片，加字后会比直接操作大型图片快。

        for i in range(int(10)):
            for j in range(int(15)):
                image = cv2AddChineseText(image, "英视睿达", (int(820/5*i), int(1280/8*j)), (0,0,255,255), 20)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image = cv2AddChineseText(image, "英视睿达", (0,0), (0, 0, 255, 255), int(100 * xsize /1280))

        # 旋转
        # x, y, c = image.shape[0:3]
        # mr = cv2.getRotationMatrix2D((x * 0.5, y * 0.5), 45, 0.5)
        # image = cv2.warpAffine(image, mr, (x, y))

        angle = -45  # 旋转角度
        center = (int(820), 0)  # 中心点
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (820, 1280), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        image = cv2.resize(image, (ysize,xsize), interpolation=cv2.INTER_NEAREST)

        return image

    data = read_tif(from_path)
    im_data = data['data']
    im_bands = data['bands']
    proj = data['projection']
    transform = data['transform']
    if proj is None:
        proj = 'GEOGCS["WGS 84",\
                     DATUM["WGS_1984",\
                             SPHEROID["WGS 84",6378137,298.257223563, \
                                    AUTHORITY["EPSG","7030"]], \
                             AUTHORITY["EPSG","6326"]], \
                     PRIMEM["Greenwich",0, \
                            AUTHORITY["EPSG","8901"]], \
                     UNIT["degree",0.0174532925199433, \
                            AUTHORITY["EPSG","9122"]],\
                     AUTHORITY["EPSG","4326"]]'
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 将(通道数、高、宽)顺序调整为(高、宽、通道数)
    # print('shape of im data:', im_data.shape)

    watermark = generate_watermark(im_data.shape[0]*2,im_data.shape[1])

   # 设置高、宽、通道数
    # if len(im_data.shape) == 4:  # 单个项元是数组
    #     im_data = im_data.reshape(im_data.shape[:3])

    im_shape = list(im_data.shape)
    im_shape.remove(im_bands)
    im_height, im_width = im_shape
    band_idx = im_data.shape.index(im_bands)
    # 找出波段是在第几个

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(to_path, im_width, im_height, im_bands, datatype)

    # if dataset is not None:
    dataset.SetGeoTransform(transform)  # 写入仿射变换参数
    dataset.SetProjection(proj)  # 写入投影

    if im_bands == 1:

        # print(im_data[:, 0,:].shape)
        if band_idx == 0:
            data = addmark_1_band(im_data[0, :, :], watermark)
            dataset.GetRasterBand(1).WriteArray(data)
        elif band_idx == 1:
            data = addmark_1_band(im_data[:, 0, :], watermark)
            dataset.GetRasterBand(1).WriteArray(data)
        elif band_idx == 2:
            data = addmark_1_band(im_data[:, :, 0], watermark)
            dataset.GetRasterBand(1).WriteArray(data)

    else:
        for i in range(im_bands):
            # if i != 0:
            #     if band_idx == 0:
            #         dataset.GetRasterBand(i + 1).WriteArray(im_data[i, :, :])
            #     elif band_idx == 1:
            #         dataset.GetRasterBand(i + 1).WriteArray(im_data[:, i, :])
            #     elif band_idx == 2:
            #         dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])
            # else:
            if band_idx == 0:
                data = addmark_m_band(im_data[i, :, :], watermark)
                dataset.GetRasterBand(i + 1).WriteArray(data)
            elif band_idx == 1:
                data = addmark_m_band(im_data[:, i, :], watermark)
                dataset.GetRasterBand(i + 1).WriteArray(data)
            elif band_idx == 2:
                data = addmark_m_band(im_data[:, :, i], watermark)
                dataset.GetRasterBand(i + 1).WriteArray(data)

    dataset.FlushCache()
    del dataset
    driver = None


def cropToCutline(input_path,output_path):
#这样操作后文件会变小，但数据形状没有变化
    ds = gdal.Warp(output_path,
                input_path,
                cropToCutline=True,
                format='GTiff',
                    )   
    ds = None

def cut_tif(origin_data,origin_transform,output_size):
    origin_size = origin_data.shape
    x = origin_transform[0]
    y = origin_transform[3]
    x_step = origin_transform[1]
    y_step = origin_transform[5]
    output_x_step = x_step# * output_size[0]/origin_size[0]
    output_y_step = y_step# * output_size[1]/origin_size[1]
    for i in range(origin_size[0]//output_size[0]):
        for j in range(origin_size[1]//output_size[1]):
            output_data = origin_data[i*output_size[0]:(i+1)*output_size[0],j*output_size[1]:(j+1)*output_size[1],:]
            output_transform = (x+j*output_x_step*output_size[0],output_x_step,0,y+i*output_y_step*output_size[0],0,output_y_step) 
#             if output_data.max() >100:
            write_tif(f'test/{i}_{j}.tif', output_data, output_transform)
