import time
from functools import partial, reduce
import csv
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

import cv2
import random
import os.path as osp
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from models.model import VQA_Network

import sys

sys.path.append('.')
sys.path.append('..')
# sys.path.append('...')
import datasets


class Trainer_for_test:
    def __init__(
            self,
            args,
            config,

    ):
        super().__init__()
        self.args = args
        self.config = config
        self.gpu_list = [int(item) for item in self.args.gpu_id.split(',')]
        self.device = torch.device("cuda:" + self.args.gpu_id.split(',')[0])
        self.build_datasets()
        self.build_models()
        self.best_results = -1, -1, -1, 1999
        self.best_results_ema = -1, -1, -1, 1999
        self.key_list = self.config['model']['type'].split(',')

    def build_models(self, ):
        self.model = VQA_Network(self.config).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_list)
        self.model2 = VQA_Network(self.config).to(self.device)
        self.model2 = torch.nn.DataParallel(self.model2, device_ids=self.gpu_list)
        if self.config["load_path"] is not None:
            state_dict = torch.load(self.config["load_path"], map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            else:
                state_dict = state_dict
        if self.config["load_path2"] is not None:
            state_dict2 = torch.load(self.config["load_path2"], map_location=self.device)
            if 'state_dict2' in state_dict2:
                state_dict2 = state_dict2['state_dict2']
            else:
                state_dict2 = state_dict2

            msg = self.model.load_state_dict(state_dict, strict=False)
            msg2 = self.model2.load_state_dict(state_dict2, strict=False)

            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"模型总参数数量: {total_params}")
            print(msg)

        if self.config["ema"]:
            from copy import deepcopy

            self.model_ema = deepcopy(self.model)
            self.model2_ema = deepcopy(self.model2)
        else:
            self.model_ema = None
            self.model2_ema = None

    def build_datasets(self):

        if 'val' in self.config["data"]:
            val_dataset = getattr(datasets, self.config["data"]["val"]["type"])(self.config["data"]["val"]["args"],
                                                                                None)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                          num_workers=self.config["num_workers"], pin_memory=True, )

    def inferece(self):

        output_results = []

        for i, data in enumerate(tqdm(self.val_loader, desc="Validating")):
            self.model.eval()
            for key in self.key_list:
                if key in data:
                    data[key] = data[key].to(self.device)
                    b, c, t, h, w = data[key].shape
                    data[key] = (
                        data[key]
                        .reshape(
                            b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                        )
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(
                            b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                        )
                    )
            with torch.no_grad():
                pred = self.model(inputs=data, reduce_scores=True).cpu().numpy()

            output_results.append((data["video_name"][0], pred.mean(0).item()))

        output_results2 = []
        for i, data in enumerate(tqdm(self.val_loader, desc="Validating")):
            self.model2.eval()
            for key in self.key_list:
                if key in data:
                    data[key] = data[key].to(self.device)
                    b, c, t, h, w = data[key].shape
                    data[key] = (
                        data[key]
                        .reshape(
                            b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                        )
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(
                            b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                        )
                    )
            with torch.no_grad():
                pred = self.model2(inputs=data, reduce_scores=True).cpu().numpy()

            output_results2.append((data["video_name"][0], pred.mean(0).item()))
            combined_results=[]
            for item in output_results:
                for item2 in output_results2:
                    if item[0] == item2[0]:
                        score =  0.6 * item2[1] + 0.4 * item[1]
                        combined_results.append((item[0],score))

        # 设置CSV文件的标题头
        headers = ['filename', 'score']
        # 打开文件进行写入操作
        with open('test_results.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            # 写入标题头
            writer.writeheader()
            for item in output_results:
                # 写入数据行
                writer.writerow({'filename': item[0], 'score': item[1]})
