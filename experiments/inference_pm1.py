#!/public/home/wangs/anaconda2/envs/pytorch/bin/python
# -*- coding: UTF-8 -*-

"""
@author:Wangs ACES
@time:2022/03/03
"""

import glob
import os
import sys
import datetime
import queue
import signal
import multiprocessing

import yaml
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from argparse import ArgumentParser
sys.path.append('/data6/wangs/pm1/stfen/baselines/')
#from extractBGFromNC import *



nXSize = 3600 
nYSize = 1800
MODEL = "STFEN"
DATASET_NAME = 'pm1'
LABEL = 'obs_pm1'
GPUS = '0'
YEAR = 2021
SPEC = 'pm1'
S_WIN = 20
T_WIN = 24
#MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
MONTHS = [1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CFG_PATH = '{0}/pm1_crossformer.py'.format(MODEL)
FEATURE_FILE= "/data6/wangs/pm1/train_generate/yaml_file/featureList_era5_merra2_2021.yaml"
MODEL_FILE = "/data6/wangs/pm1/stfen/experiments/checkpoints/STFEN_50/d23baabc6037b303a3b876722d44873a/STFEN_best_val_MAE.pt"
OUTDIR = "/data6/wangs/pm1/pred_fill_pm1_daily_stfen"
MEAN_STD_DF = pd.read_csv("/data6/wangs/pm1/dl/mean_std.csv", delimiter=',', index_col='var')

FEATURE_LT = [
        'BCSMASS', 'OCSMASS', 'SO2SMASS', 'SO4SMASS',
        'DMSSMASS', 'DUSMASS25', 'SSSMASS', 'TOTEXTTAU',
        'EVAP', 'RH', 'DEWP2', 'TEMP2', 'VWIND10', 'UWIND10',
        'SP', 'SSRD', 'ALBEDO', 'TPREC', 'BLH', 'TCLOUD'
        ]

def tif2nc(tiffile, SPEC):
    tem_ds = gdal.Open(tiffile)
    tem_arr = tem_ds.ReadAsArray()

    nXSize = tem_ds.RasterXSize
    nYSize = tem_ds.RasterYSize
    Bands = 1

    ds_out = xr.Dataset.from_dict({
        "coords": {
            "lat": {"dims": ("lat",),
                    "attrs": {"standard_name": "latitude", "units": "0.01 degrees", "axis": "Y"},
                    "data": np.linspace(34.0, 41.5, nYSize), },
            "lon": {"dims": ("lon",),
                    "attrs": {"standard_name": "longitude", "units": "0.01 degrees", "axis": "X"},
                    "data": np.linspace(111.5, 120.0, nXSize), },
            "time": {"dims": ("time",),
                     "attrs": {"standard_name": "time", "long_name": "Time hourly"},
                     "data": pd.date_range('2020-01-01-00', periods=Bands, freq='H')}},
        "data_vars": {SPEC: {"dims": ("time", "lat", "lon"),
                             "data": np.zeros((Bands, nYSize, nXSize), dtype='float32')}}
    })
    # print(ds_out[SPEC].shape)
    #ds_out[SPEC][0, :, :] = np.flipud(tem_arr)
    ds_out[SPEC][0, :, :] = tem_arr

    # ds_out.to_netcdf('/public/home/wangs/propy_Win/train_generate/ocecNCP/%s.nc' % SPEC, mode='w',
    #                 format="NETCDF3_CLASSIC", encoding={SPEC: {'zlib': True}}, engine='netcdf4')

    return ds_out


def normalize_arr(arr, MEAN_STD_DF, var_lt):
    """
    arr: [times, variables, xx, yy]
    """
    for i in range(arr.shape[1]):
        arr[:,i,:,:] = (arr[:,i,:,:] - MEAN_STD_DF.loc[var_lt[i],'mean']) / MEAN_STD_DF.loc[var_lt[i],'std']
    return arr


def normalize_df(df, MEAN_STD_DF, var_lt):
    var_lt_label = var_lt.copy()
    var_lt_label.append(LABEL)
    for i in range(len(var_lt_label)):
        spec = var_lt_label[i]
        mean = MEAN_STD_DF.loc[spec, 'mean']#.values[0]
        std = MEAN_STD_DF.loc[spec, 'std']#.values[0]
        print(mean)
        df[spec] = (df[spec] - mean) / std
    return df


def import_config(path: str, verbose: bool = True):
    """Import config by path

    Examples:
        ```
        cfg = import_config('config/my_config.py')
        ```
        is equivalent to
        ```
        from config.my_config import CFG as cfg
        ```

    Args:
        path (str): Config path
        verbose (str): set to ``True`` to print config

    Returns:
        cfg (Dict): `CFG` in config file
    """

    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.').replace('\\', '.')
    cfg_name = path.split('.')[-1]
    cfg = __import__(path, fromlist=[cfg_name]).CFG

    # if verbose:
    #     print(config_str(cfg))
    return cfg


def define_model(cfg) -> nn.Module:
    return cfg["MODEL"]["ARCH"](**cfg.MODEL.PARAM)


def extractFromNetCDF(f_lt, var_lt, time, S_WIN,):
    """
    This function will extract the data from given datetime
    :return a dataframe
    """

    for i in range(len(f_lt)):
        infile = f_lt[i]
        #print(infile)
        SPEC = var_lt[i]

        if '.tif' in infile:
            # input_tif = gdal.Open(infile)
            # dataset = input_tif.ReadAsArray()
            input_nc = tif2nc(infile, SPEC)
            dataset = input_nc[SPEC]
        elif '.nc' in infile:
            input_nc = xr.open_dataset(infile, engine='netcdf4')
            var_list = np.array(input_nc.data_vars.variables)
            dataset = input_nc[var_list[0]]
        else:
            sys.exit('file not found')

        if dataset.shape[0] != 1:
            value_lt = dataset.loc[time].values
        else:
            value_lt = dataset[0].values
            # normaliaze
            value_lt = (value_lt - MEAN_STD_DF.loc[var_lt[i],'mean']) / MEAN_STD_DF.loc[var_lt[i],'std']
            value_lt = value_lt.astype("float16")
        print(value_lt.shape)

        pad_width = [(0, S_WIN - 1), (0, S_WIN - 1)]
        padded_arr = np.pad(value_lt, pad_width, mode='wrap')
        slid_arr = np.lib.stride_tricks.sliding_window_view(padded_arr,(S_WIN,S_WIN))
        print("slid_arr: ", slid_arr.shape)

        if i == 0:
            Bands = dataset.shape[0]
            #print('Bands:', Bands)
            yy = np.arange(0, dataset.shape[1])
            xx = np.arange(0, dataset.shape[2])
            xx_yy = np.meshgrid(xx, yy)
            print(xx_yy[0].shape)
            local_arr = np.concatenate(([xx_yy[0]], [xx_yy[1]], [value_lt]), axis=0)
            region_arr = slid_arr.reshape(-1,S_WIN,S_WIN)
        else:
            local_arr = np.concatenate((local_arr, [value_lt]), axis=0)
            region_arr = np.concatenate((region_arr, slid_arr.reshape(-1,S_WIN,S_WIN)), axis=0)

    combine_df = pd.DataFrame(local_arr.reshape(local_arr.shape[0], -1).T, columns=['xx','yy'] + var_lt)

    combine_df['time'] = time

    assert combine_df.shape[0] == region_arr.shape[0]

    return combine_df, region_arr


def modelPrediction(cfg, SPEC, time_lt, f_lt, var_lt, outfile, MODEL_FILE, S_WIN,T_WIN):

    """
    This script will generate the prediction using MODEL_FILE
    the time_lt should be datatime dtype
    """
    try:
        model = torch.load(MODEL_FILE)
        model = model.to(device)
        model.eval()  # Set model to evaluate mode
    except:
        checkpoint = torch.load(MODEL_FILE)
        model = define_model(cfg=cfg)
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        model.eval()  # Set model to evaluate mode

    print(model)

    Bands = len(time_lt)

    for i in range(len(time_lt)):
        time = time_lt[i]
        combine_df, region_arr = extractFromNetCDF(f_lt, var_lt, time, S_WIN,)

        region_arr = normalize_arr(region_arr)
        combine_df = normalize_df(combine_df)

        print(region_arr.shape)
        print(combine_df.shape)

        #Rolling features
        history_arr = combine_df.copy()[var_lt]
        pad_width = [(0, T_WIN - 1)]
        padded_arr = np.pad(history_arr, pad_width, mode='wrap')
        history_arr = np.lib.stride_tricks.sliding_window_view(padded_arr,(T_WIN))
        print(history_arr.shape)
        
        # if time < datetime.date(YEAR, 1, T_WIN + 1):
        # else:
        #     for jj in range(T_WIN-1):
        #         df_tmp = extractFromNetCDF(f_lt, varlt11, time - datetime.timedelta(hours=-(jj+1)), S_WIN, w)
        #         #df_tmp = extractLocalFromNetCDF(f_lt, var_lt, time)
        #         history_arr = history_arr + df_tmp[var_lt]
        #     # 平均加权？？？
        #     history_arr = history_arr / T_WIN
        #     for fea_roll in var_lt:
        #         combine_df['roll' + str(T_WIN) + 'gs_' + fea_roll] = df_roll[fea_roll]
                    
        df = combine_df.fillna(method='ffill')

        # local feature data
        local_df = df[FEATURE_LT]
 
        print('local_df shape:')
        print(local_df.shape)
        #print(local_df.columns.values)

        # load model to predict
        if i == 0:
            model = joblib.load(filename=MODEL_FILE)

            ds_out = xr.Dataset.from_dict({
            "coords": {
            "lat": {"dims": ("lat",),
                    "attrs": {"standard_name": "latitude", "units": "0.1 degrees", "axis": "Y"},
                    "data": np.linspace(5, 40, nYSize), },
            "lon": {"dims": ("lon",),
                    "attrs": {"standard_name": "longitude", "units": "0.1 degrees", "axis": "X"},
                    "data": np.linspace(60, 100, nXSize), },
            "time": {"dims": ("time",),
                     "attrs": {"standard_name": "time", "long_name": "Time hourly"},
                     "data": pd.date_range(time_lt[0], periods=Bands, freq='H')}},
            "data_vars": {SPEC: {"dims": ("time", "lat", "lon"),
                                 "data": np.zeros((Bands, nYSize, nXSize), dtype='float32')}}})

        with torch.no_grad():
            outputs = model((history_arr, local_df.values, region_arr))
        #ds_out[SPEC][i, :, :] = y_pred.reshape(nYSize, nXSize)
        ds_out[SPEC][i, :, :] = np.flipud(outputs.reshape(nYSize, nXSize))

        ds_out.to_netcdf(outfile, mode='w', format="NETCDF3_CLASSIC", encoding={SPEC: {'zlib': True}}, engine='netcdf4')


def manual_fct2(job_queue, result_queue, cfg, SPEC, time_lt, f_lt, var_lt, outfile, MODEL_FILE, S_WIN,T_WIN):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while not job_queue.empty():
        try:
            job = job_queue.get(block=False)
            result_queue.put(modelPrediction(cfg, SPEC, time_lt, f_lt, var_lt, outfile, MODEL_FILE, S_WIN,T_WIN))
        except queue.Empty:
            pass


if __name__ == '__main__':

    cfg = import_config(CFG_PATH)

    mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    time_list = [pd.date_range('%d-%s-01' % (YEAR, str(k).zfill(2)), periods=mdays[k-1], freq='D') for k in MONTHS]

    parser = ArgumentParser(description='Inference by STFEN')
    #parser.add_argument('-m', '--model', default=MODEL_NAME, help='model name')
    parser.add_argument('-d', '--dataset', default=DATASET_NAME, help='dataset name')
    parser.add_argument('-g', '--gpus', default=GPUS, help='visible gpus')
    parser.add_argument('-y', '--YEAR', default=YEAR, help='YEARs for inference')
    args = parser.parse_args()

    with open(FEATURE_FILE) as f:
        yamldic = yaml.load(f, Loader=yaml.FullLoader)

    var_lt = yamldic['featuresName']
    f_lt = yamldic['fileList']

    # aligned features and path lists
    lt_tmp = []
    for i in range(len(var_lt)):
        if var_lt[i] not in FEATURE_LT:
            lt_tmp.append(i)
    counter = 0
    for idx in lt_tmp:
        idx = idx - counter
        var_lt.pop(idx)
        f_lt.pop(idx)
        counter += 1
    print(var_lt)
    print(f_lt)

    if not (os.path.exists(OUTDIR)):
        os.mkdir(OUTDIR)

    core_num = 1 #len(time_list)
    print('core num:',core_num)
    job_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    for i in range(core_num):
        job_queue.put(None)

    workers = []
    for i in range(core_num):
        time_lt = time_list[i]
        month = time_lt[0].month

        #######For Test########
        indx = [kk for kk in range(1,5)]
        time_lt = time_lt[indx]
        #######For Test########

        outfile = os.path.join(OUTDIR, '%s_arr_10km_%d%d.nc' % (SPEC, YEAR, month))

        p = multiprocessing.Process(target=manual_fct2,
                                    args=(job_queue, result_queue, cfg, SPEC, time_lt, f_lt, var_lt, outfile, MODEL_FILE, S_WIN,T_WIN))
        p.start()
        workers.append(p)
    try:
        for worker in workers:
            worker.join()
            print(worker)
    except KeyboardInterrupt:
        print('parent fun received ctrl-c')

        for worker in workers:
            worker.terminate()
            worker.join()

    while not result_queue.empty():
        print(result_queue.get(block=False))
        

