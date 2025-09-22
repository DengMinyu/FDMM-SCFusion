import os
import sys
import glob
import time

import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch import einsum
from model.Network import Network
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
import torch.nn.functional as F


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class Fusion:
    def __init__(self,
                 modelpath='best_network.pth',
                 dataroot='./Datasets/Eval/',
                 # 包含所有要测试的数据集名称
                 dataset_names=['Lytro', 'MFFW', 'MFI-WHU', 'Grayscale'],
                 threshold=0.0015,
                 window_size=5,
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAMES = dataset_names  # 多个数据集名称
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)
        # 加载一次模型供所有数据集使用
        self.model = self.LoadWeights(self.MODELPATH)

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAMES and len(self.DATASET_NAMES) > 0:
            # 遍历所有数据集
            for dataset_name in self.DATASET_NAMES:
                print(f"\n开始处理数据集: {dataset_name}")
                self.DATASET_NAME = dataset_name
                self.SAVEPATH = '/' + self.DATASET_NAME
                self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
                EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
                self.FusionProcess(self.model, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("需要指定测试数据集!")

    def LoadWeights(self, modelpath):
        model = Network().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()

        from thop import profile, clever_format
        flops, params = profile(model, inputs=(torch.rand(1, 3, 520, 520).cuda(), torch.rand(1, 3, 520, 520).cuda()))
        flops, params = clever_format([flops, params], "%.5f")
        print('flops: {}, params: {}\n'.format(flops, params))
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = Consistency.Binarization(img_tensor)
        if threshold != 0:
            Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results' + savepath):
            os.makedirs('./Results' + savepath)

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []

        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()
                D = model(A, B)
                D = torch.where(D[0] > 0.5, 1., 0.)
                D = self.ConsisVerif(D, threshold)

                D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()
                A = cv2.imread(eval_list_A[cnt - 1])
                B = cv2.imread(eval_list_B[cnt - 1])
                IniF = A * D + B * (1 - D)
                cv2.imwrite(f'./Results{savepath}/{self.DATASET_NAME}-{str(cnt).zfill(2)}.png', IniF)

                cnt += 1
                running_time.append(time.time() - start_time)

        # 计算并打印运行时间统计
        running_time_total = 0
        for i in range(len(running_time)):
            print(f"处理时间 {i + 1}: {running_time[i]:.6f} s")
            if i != 0:  # 排除第一个样本的时间（可能包含初始化开销）
                running_time_total += running_time[i]

        if len(running_time) > 1:
            print(f"\n平均处理时间: {running_time_total / (len(running_time) - 1):.6f} s")
        print(f"结果已保存至: ./Results{savepath}\n")


if __name__ == '__main__':
    # 创建Fusion实例，会自动处理所有指定的数据集
    f = Fusion()
    f()
