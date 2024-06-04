import os
import sys

# TODO: remove it when basicts can be installed by pip
#sys.path.append(os.path.abspath(__file__ + "/../../.."))
sys.path.append(os.path.abspath(__file__ + "/../.."))
from easydict import EasyDict
from basicts.losses import masked_mae, masked_mse, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.data import SpatialTimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner

from .arch import STFEN, TFEN

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STFEN model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = SpatialTimeSeriesForecastingDataset
#CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "pm1"
CFG.DATASET_TYPE = "pm1"
CFG.DATASET_INPUT_LEN = 24
CFG.DATASET_OUTPUT_LEN = 1
CFG.GPU_NUM = 0
#CFG.IMG_DIR = "/data6/wangs/pm1/dl/img_data_w40_concat"
CFG.IMG_DIR = "/data6/wangs/pm1/dl/img_data_w40_concat_npy" # 图像所在目录

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "TFEN"
CFG.MODEL.ARCH = TFEN
NUM_NODES = 20
CFG.MODEL.PARAM = {
    "femodel": "vgg", #"resnet", # 图像特征提取网络名称
    "tsmodel": "patchtst", #, # 时序特征提取网络名称
    "img_dim": 21, #图像通道数（维度）
    "fe_dim": 32, #图像特征提取输出维度
    "num_layer": 3, # MLP 层数
    "num_nodes": NUM_NODES,

    "enc_in": NUM_NODES,                        # num nodes
    "seq_len": CFG.DATASET_INPUT_LEN,           # input sequence length
    "pred_len": CFG.DATASET_OUTPUT_LEN,         # prediction sequence length
    "e_layers": 3,                              # num of encoder layers
    "n_heads": 16,
    "d_model": 128,
    "d_ff": 256,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "patch_len": 14,
    "stride": 8,
    "individual": 0,                            # individual head; True 1 False 0
    "padding_patch": "end",                     # None: None; end: padding on the end
    "revin": 1,                                 # RevIN; True 1 False 0
    "affine": 0,                                # RevIN-affine; True 1 False 0
    "subtract_last": 0,                         # 0: subtract mean; 1: subtract last
    "decomposition": 0,                         # decomposition; True 1 False 0
    "kernel_size": 25,                          # decomposition-kernel
}

CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_rmse #masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.01,
    "weight_decay": 0.0001,
    #"lr":0.005,
    #"momentum":0.9,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 5, 10, 15, 20, 25, 30, 35, 40],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "/data6/wangs/pm1/stfen/" + "datasets/" + CFG.DATASET_NAME + "_mini"
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64 #2048#64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = False#True
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "/data6/wangs/pm1/stfen/" + "datasets/" + CFG.DATASET_NAME + "_mini"
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64 #1024#16
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 8
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "/data6/wangs/pm1/stfen/" + "datasets/" + CFG.DATASET_NAME + "_mini"
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64 #64#1024#16
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 8
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
# CFG.EVAL.HORIZONS = [12, 24, 48, 96, 192, 288, 336]
CFG.EVAL.HORIZONS = [1]
