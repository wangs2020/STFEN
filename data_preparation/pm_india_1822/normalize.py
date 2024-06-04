#!/public/home/wangs/anaconda2/envs/pytorch/bin/python
# -*- encoding: utf-8 -*-

import sys
import json
import os
import glob
import time
import yaml
import numpy as np
import pandas as pd
import xarray as xr


save_img_lable = True
calcu_mean_std = False#True 
nc_mean_std = True

year = 1822
spec = "pm25"
label = "obs_%s" % spec
dataname = "pm_india_1822"

source_dir = "/data9/wangs/india/train_generate_4longTerm/pm_train_4cnn_single_40_1822_daily/"
#nc_dir = "/data6/wangs/pm1/stfen/datasets/{0}".format(dataname)
nc_dir = "/data6/wangs/pm1/dataset_npy/{0}".format(dataname)
stfile = "/data9/wangs/india/obs_process/stations.xy_rectangle.csv"
yamlfile = "/data9/wangs/india/train_generate_4longTerm/yaml_file/featureList_PM_era5_merra2_1822.yaml" 
local_file = "/data9/wangs/india/train_generate_4longTerm/output_era5_merra2/pm25_train_era5_merra2_daily_1822.csv"
outfile = "/data6/wangs/pm1/stfen/datasets/raw_data/{0}/{0}.csv".format(dataname)
mean_std_file = "./mean_std.csv"

if not os.path.exists(nc_dir):
    os.makedirs(nc_dir)

fea_lt = [
    #'ALBEDO',
    'SP','VWIND10','UWIND10','SSRD','TCLOUD',
    'EVAP','BLH','TPREC','TEMP2','DEWP2',
    'BCSMASS','OCSMASS','DUSMASS','SO2SMASS','SO4SMASS',
    'TOTEXTTAU',
    #'DUSMASS25',
    #'dayofyear','month',
]
with open(yamlfile) as f:
    yamldic = yaml.load(f, Loader=yaml.FullLoader)
var_lt = yamldic['featuresName']
f_lt = yamldic['fileList']

img_labels = pd.read_csv(local_file, header=0, sep=',')
st_df = pd.read_csv(stfile, header=0, sep=',')

img_labels = pd.merge(st_df['st_ID'], img_labels, on=["st_ID"], how='inner')
img_labels = img_labels.drop_duplicates(subset=['st_ID','time'])
img_labels.rename(columns={'value': label, 'time': 'date'},inplace=True)
img_labels = img_labels.loc[img_labels[label] > 0, :]
img_labels = img_labels.dropna()

# cut outlier
qua_df = img_labels.quantile([.001, .999], numeric_only=True)
lowcut = qua_df[label].iloc[0]
highcut = qua_df[label].iloc[1]
img_labels = img_labels[img_labels[label] < highcut]
img_labels = img_labels[img_labels[label] > lowcut]
print(lowcut)
print(highcut)

print(st_df.info())
print(img_labels.info())

# aligned features and path lists
lt_tmp = []
for i in range(len(var_lt)):
    if var_lt[i] not in fea_lt:
        lt_tmp.append(i)
counter = 0
for idx in lt_tmp:
    idx = idx - counter
    var_lt.pop(idx)
    f_lt.pop(idx)
    counter += 1
print(var_lt)
print(f_lt)

# calcut labels mean and std
if calcu_mean_std:
    label_mean = img_labels[label].mean()
    label_std = img_labels[label].std()
    img_labels[label] = (img_labels[label] - label_mean) / label_std

    mean_std_df = pd.DataFrame(columns=[['var', 'mean', 'std']])
    mean_std_df.loc[len(mean_std_df)] = [label, label_mean, label_std]
    for i in range(len(f_lt)):
        f0 = f_lt[i]
        spec = var_lt[i]
        print(spec)
        input_nc = xr.open_dataset(f0, engine='netcdf4', )
        var_list = np.array(input_nc.data_vars.variables)
        dataset = input_nc[var_list[0]]
        #if spec in ['BCSMASS', 'OCSMASS', 'SO2SMASS', 'SO4SMASS', 'DUSMASS25', 'SSSMASS25', 'SSSMASS', 'DMSSMASS']:
        #    dataset *= 10e9
        mean = dataset.mean().values
        std = dataset.std().values
        mean_std_df.loc[len(mean_std_df)] = [spec, mean, std]
        print(mean_std_df)
    mean_std_df.to_csv(mean_std_file, index=False)

var_lt_label = var_lt.copy()
var_lt_label.append(label)
mean_std_df = pd.read_csv(mean_std_file, delimiter=',', index_col='var')
for i in range(len(var_lt_label)):
    spec = var_lt_label[i]
    print(spec)
    mean = mean_std_df.loc[spec, 'mean']#.values[0]
    std = mean_std_df.loc[spec, 'std']#.values[0]
    print(mean)
    img_labels[spec] = (img_labels[spec] - mean) / std

if save_img_lable:
    print(img_labels.info())
    img_labels.to_csv(outfile, index=False, float_format='%.3f')

if nc_mean_std:
    st_df = pd.read_csv(stfile)
    for idx, row in st_df.iterrows():
        st = row['st_ID']
        f0 = os.path.join(source_dir, str(st)+".nc")
        if os.path.exists(f0):
            input_nc = xr.open_dataset(f0, engine='netcdf4') 
            for spec in var_lt:
                input_nc[spec] = (input_nc[spec] - mean_std_df.loc[spec,'mean']) / mean_std_df.loc[spec,'std']
            #input_nc.to_netcdf(os.path.join(nc_dir, str(st)+".nc"), mode='w', format="NETCDF4", encoding={spec: {'zlib': True}}, engine='netcdf4')
            np.save(file=os.path.join(nc_dir, str(st)+".npy"), arr=input_nc[var_lt].to_array().values.astype(np.float16))
        else:
            print(f'{f0} not found')
            #img_labels = img_labels.loc[img_labels['st_ID'] != st,:]
            img_labels.drop(img_labels[img_labels['st_ID'] == st].index, inplace=True)

if save_img_lable:
    #img_labels.rename({'time': 'date'}).to_csv(outfile, index=False, float_format='%.3f')
    img_labels.to_csv(outfile, index=False, float_format='%.3f')
