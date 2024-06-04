#!/public/home/wangs/anaconda2/envs/pytorch/bin/python
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../.."))

import sys
import torch
from argparse import ArgumentParser

from basicts import launch_training

torch.set_num_threads(3) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")

    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_informer.py", help="training config")
   # parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_crossformer.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_patchtst.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_linear.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_dlinear.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_patchtst.py", help="training config")

    # for ablation study
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_stfen.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_tfen.py", help="training config")
    parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_sfen.py", help="training config")
    #parser.add_argument("-c", "--cfg", default="baselines/STFEN/pm1_nfen.py", help="training config")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("torch.cuda.device_count()=",torch.cuda.device_count())
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    launch_training(args.cfg, args.gpus)
