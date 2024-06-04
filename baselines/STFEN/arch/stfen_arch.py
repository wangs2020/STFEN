import torch
from torch import nn
from torchvision import models
import os
import sys
sys.path.append(os.path.abspath(__file__ + "../../"))
from .mlp import MLP
from STID.arch import STID
from Informer.arch import Informer
from DLinear.arch import DLinear
from NLinear.arch import NLinear
from PatchTST.arch import PatchTST
from Linear.arch import Linear
from Crossformer.arch import Crossformer
import inspect


def initialize_fe_model(model_name, in_channels, fe_dim, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_fe = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_fe = models.resnet18(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_fe, feature_extract)
        #model_fe.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model_fe.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        num_ftrs = model_fe.fc.in_features
        model_fe.fc = nn.Linear(num_ftrs, fe_dim)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_fe = models.alexnet(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_fe, feature_extract)
        model_fe.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
        num_ftrs = model_fe.classifier[6].in_features
        model_fe.classifier[6] = nn.Linear(num_ftrs, fe_dim)
        input_size = 224
    elif model_name == "vgg":
        """ Vgg
        """
        model_fe = models.vgg11_bn(pretrained=use_pretrained)
        #set_parameter_requires_grad(self.emb, self.feature_extract)
        model_fe.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model_fe.features = model_fe.features[:-8]
        num_ftrs = model_fe.classifier[6].in_features
        model_fe.classifier[6] = nn.Linear(num_ftrs, fe_dim)
    elif model_name == "densenet":
        """"densenet"
        """
        model_fe = models.densenet121(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_fe, feature_extract)
        num_ftrs = model_fe.classifier.in_features
        model_fe.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    """
    TODO: 补充其他特征提取模型
    """

    return model_fe


def initialize_ts_model(model_name, model_args):
    
    if model_name == "stid":
        model_ts = STID(**model_args)

    elif model_name == "informer":
        constructor_params = inspect.signature(Informer).parameters
        valid_params = {param: model_args[param] for param in constructor_params if param in model_args}
        model_ts = Informer(**valid_params)

    elif model_name == "dlinear":
        model_ts = DLinear(**model_args)

    elif model_name == "nlinear":
        model_ts = NLinear(**model_args)

    elif model_name == "patchtst":
        constructor_params = inspect.signature(PatchTST).parameters
        valid_params = {param: model_args[param] for param in constructor_params if param in model_args}
        model_ts = PatchTST(**valid_params)
    
    elif model_name == "crossformer":
        constructor_params = inspect.signature(Crossformer).parameters
        valid_params = {param: model_args[param] for param in constructor_params if param in model_args}
        model_ts = Crossformer(**valid_params)
    return model_ts
    

class STFEN(nn.Module):
    """
    Paper:
    Link:
    Official Code:
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.model_args = model_args
        self.femodel = model_args["femodel"]
        self.tsmodel = model_args["tsmodel"]
        self.img_dim = model_args["img_dim"]
        self.fe_dim = model_args["fe_dim"]
        self.num_nodes = model_args["num_nodes"]
        self.num_layer = model_args["num_layer"]


        # feature extract
        self.fe_emb = initialize_fe_model(self.femodel, self.img_dim, self.fe_dim)
        self.ts_emb = initialize_ts_model(self.tsmodel, model_args)

        #decoder
        # TODO:MLP网络存在冗余，改为用nn.Sequential直接串起来
        self.cat_dim = self.num_nodes + self.num_nodes + self.fe_dim
        self.decoder = nn.Sequential(
            *[MLP(self.cat_dim, self.cat_dim) for _ in range(self.num_layer)])

        self.linear_layer = nn.Linear(in_features=self.cat_dim, out_features=1)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, current_data, img, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # regression
        time_emb = self.ts_emb(history_data, current_data, batch_seen, epoch, train)

        # feature extract for img
        img_emb = self.fe_emb(img.to(torch.float32))
        # print(current_data.shape)
        # print(time_emb.shape)
        # print(img_emb.shape)
        cat_data = torch.cat((current_data[:,:,:,-1].squeeze(1), time_emb.squeeze(), img_emb), dim=1)
        #print(cat_data.shape)
        prediction = self.decoder(cat_data)

        out = self.linear_layer(prediction).unsqueeze(-1)
        #print(out.shape)

        return out


class TFEN(nn.Module):
    """
    Paper:
    Link:
    Official Code:
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.model_args = model_args
        self.femodel = model_args["femodel"]
        self.tsmodel = model_args["tsmodel"]
        self.img_dim = model_args["img_dim"]
        self.fe_dim = model_args["fe_dim"]
        self.num_nodes = model_args["num_nodes"]
        self.num_layer = model_args["num_layer"]


        # feature extract
        self.fe_emb = initialize_fe_model(self.femodel, self.img_dim, self.fe_dim)
        self.ts_emb = initialize_ts_model(self.tsmodel, model_args)

        #decoder
        # TODO:MLP网络存在冗余，改为用nn.Sequential直接串起来
        self.cat_dim = self.num_nodes + self.num_nodes #+ self.fe_dim
        self.decoder = nn.Sequential(
            *[MLP(self.cat_dim, self.cat_dim) for _ in range(self.num_layer)])

        self.linear_layer = nn.Linear(in_features=self.cat_dim, out_features=1)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, current_data, img, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # regression
        time_emb = self.ts_emb(history_data, current_data, batch_seen, epoch, train)
        cat_data = torch.cat((current_data[:,:,:,-1].squeeze(1), time_emb.squeeze()), dim=1)
        prediction = self.decoder(cat_data)
        out = self.linear_layer(prediction).unsqueeze(-1)

        return out
    

class SFEN(nn.Module):
    """
    Paper:
    Link:
    Official Code:
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.model_args = model_args
        self.femodel = model_args["femodel"]
        self.tsmodel = model_args["tsmodel"]
        self.img_dim = model_args["img_dim"]
        self.fe_dim = model_args["fe_dim"]
        self.num_nodes = model_args["num_nodes"]
        self.num_layer = model_args["num_layer"]


        # feature extract
        self.fe_emb = initialize_fe_model(self.femodel, self.img_dim, self.fe_dim)
        self.ts_emb = initialize_ts_model(self.tsmodel, model_args)

        #decoder
        # TODO:MLP网络存在冗余，改为用nn.Sequential直接串起来
        self.cat_dim = self.num_nodes + self.fe_dim
        self.decoder = nn.Sequential(
            *[MLP(self.cat_dim, self.cat_dim) for _ in range(self.num_layer)])

        self.linear_layer = nn.Linear(in_features=self.cat_dim, out_features=1)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, current_data, img, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # feature extract for img
        img_emb = self.fe_emb(img.to(torch.float32))
        cat_data = torch.cat((current_data[:,:,:,-1].squeeze(1), img_emb), dim=1)
        prediction = self.decoder(cat_data)
        out = self.linear_layer(prediction).unsqueeze(-1)

        return out


class NFEN(nn.Module):
    """
    Paper:
    Link:
    Official Code:
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.model_args = model_args
        self.femodel = model_args["femodel"]
        self.tsmodel = model_args["tsmodel"]
        self.img_dim = model_args["img_dim"]
        self.fe_dim = model_args["fe_dim"]
        self.num_nodes = model_args["num_nodes"]
        self.num_layer = model_args["num_layer"]


        # feature extract
        self.fe_emb = initialize_fe_model(self.femodel, self.img_dim, self.fe_dim)
        self.ts_emb = initialize_ts_model(self.tsmodel, model_args)

        #decoder
        # TODO:MLP网络存在冗余，改为用nn.Sequential直接串起来
        self.cat_dim = self.num_nodes
        self.decoder = nn.Sequential(
            *[MLP(self.cat_dim, self.cat_dim) for _ in range(self.num_layer)])

        self.linear_layer = nn.Linear(in_features=self.cat_dim, out_features=1)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, current_data, img, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # feature extract for img
        #img_emb = self.fe_emb(img.to(torch.float32))
        cat_data = current_data[:,:,:,-1].squeeze(1)
        prediction = self.decoder(cat_data)
        out = self.linear_layer(prediction).unsqueeze(-1)

        return out
