import os

#import xarray as xr
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ...utils import load_pkl


class SpatialTimeSeriesForecastingDataset(Dataset):
    """Spatial Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, img_file_name: str, img_file_dir: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path, img_file_name)

        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()

        # read index
        self.index = load_pkl(index_file_path)[mode]

        # load dataframe
        #self.img_name = pd.read_csv(img_file_name, header=0, sep=',')
        self.img_name = load_pkl(img_file_name)
        self.doy = pd.to_datetime(self.img_name['date']).dayofyear
        self.st_lt = self.img_name['st_ID']
        

        self.img_file_dir = img_file_dir

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str, img_labels_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        if not os.path.isfile(img_labels_file_path):
            raise FileNotFoundError("BasicTS can not find img file {0}".format(img_labels_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])

        # continuous index
        # history_data = self.data[idx[0]:idx[1]]
        # future_data = self.data[idx[1]:idx[2]]
        history_data = self.data[idx[0]:idx[1], :-1, :]
        future_data = self.data[idx[1]:idx[2], -1, :]
        current_data = self.data[idx[1]:idx[2], :-1, :]

        # add regional data
        #datee = str(self.img_name['date'][idx[1]])
        #f0 = os.path.join(self.img_name['img_path'][idx[1]])
        # dataset = xr.open_dataset(f0, engine='netcdf4').loc[dict(time=datee)]
        # out_img = dataset.to_array().values
        # out_img = np.nan_to_num(dataset.to_array().values)
        doy = int(self.doy[idx[1]]) # 获取一年中第几天
        st = self.st_lt[idx[1]] #获取idx这条数据对应的站点名称
        f0 = os.path.join(self.img_file_dir, str(st)+".npy") #img_file_dir图像目录，结合站点名得到图像路径
        out_img = np.load(f0)[:,doy-1,:,:] #索引对应的doy这一天的数据
        #TODO: 考虑按天存储数据，应该读取更快！！
        return future_data, history_data, current_data, out_img


    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)



class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, img_file_name: str, img_file_dir: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path, img_file_name)

        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()

        # read index
        self.index = load_pkl(index_file_path)[mode]

        # load dataframe
        #self.img_name = pd.read_csv(img_file_name, header=0, sep=',')
        self.img_name = load_pkl(img_file_name)
        self.doy = pd.to_datetime(self.img_name['date']).dayofyear
        self.st_lt = self.img_name['st_ID']
        

        self.img_file_dir = img_file_dir

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str, img_labels_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        if not os.path.isfile(img_labels_file_path):
            raise FileNotFoundError("BasicTS can not find img file {0}".format(img_labels_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])

        # continuous index
        history_data = self.data[idx[0]:idx[1], :-1, :]
        future_data = self.data[idx[1]:idx[2], -1, :]
        current_data = self.data[idx[1]:idx[2], :-1, :]

        # add regional data
        # doy = int(self.doy[idx[1]]) # 获取一年中第几天
        # st = self.st_lt[idx[1]] #获取idx这条数据对应的站点名称
        # f0 = os.path.join(self.img_file_dir, str(st)+".npy") #img_file_dir图像目录，结合站点名得到图像路径
        # out_img = np.load(f0)[:,doy-1,:,:] #索引对应的doy这一天的数据
        out_img = np.array([0])
        # #TODO: 考虑按天存储数据，应该读取更快！！
        return future_data, history_data, current_data


    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)


class SpatialForecastingDataset(Dataset):
    """Spatial Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, img_file_name: str, img_file_dir: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path, img_file_name)

        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()

        # read index
        self.index = load_pkl(index_file_path)[mode]

        # load dataframe
        #self.img_name = pd.read_csv(img_file_name, header=0, sep=',')
        self.img_name = load_pkl(img_file_name)
        self.doy = pd.to_datetime(self.img_name['date']).dayofyear
        self.st_lt = self.img_name['st_ID']
        

        self.img_file_dir = img_file_dir

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str, img_labels_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        if not os.path.isfile(img_labels_file_path):
            raise FileNotFoundError("BasicTS can not find img file {0}".format(img_labels_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])

        # continuous index
        #history_data = self.data[idx[0]:idx[1], :-1, :]
        future_data = self.data[idx[1]:idx[2], -1, :]
        current_data = self.data[idx[1]:idx[2], :-1, :]

        # add regional data
        doy = int(self.doy[idx[1]]) # 获取一年中第几天
        st = self.st_lt[idx[1]] #获取idx这条数据对应的站点名称
        f0 = os.path.join(self.img_file_dir, str(st)+".npy") #img_file_dir图像目录，结合站点名得到图像路径
        out_img = np.load(f0)[:,doy-1,:,:] #索引对应的doy这一天的数据
        #TODO: 考虑按天存储数据，应该读取更快！！
        return future_data, current_data, out_img


    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)

