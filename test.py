import torch
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import pickle
import math
import yaml
from collections import OrderedDict

from functools import reduce
from thop import profile
import copy
import os 

from trainer_for_test import Trainer_for_test

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="config/kwai_simpleVQA_test.yml", help="the option file"
    )

    parser.add_argument(
        "-t", "--target_set", type=str, default="val", help="target_set"
    )
    parser.add_argument('--gpu_id',type=str,default='1,2,3,4')


    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    trainer=Trainer_for_test(args, opt)
    trainer.inferece()

if __name__ == "__main__":
    main()
