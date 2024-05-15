from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        """
        - name: 数据集名称
        - root: 数据集根目录
        - mode: 数据集模式(train_l、train_u、val)
        - size: 图片尺寸
        - id_path: 数据集ID文件路径
        - nsample: 样本数量(仅在mode为train_l时有效)
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == "train_l" or mode == "train_u":  # 如果是训练模式
            with open(id_path, "r") as f:  # 从指定的路径读取ID
                self.ids = f.read().splitlines()  # 读取所有行
            if mode == "train_l" and nsample is not None:  # 如果是有标签的训练模式且指定了样本数
                self.ids *= math.ceil(nsample / len(self.ids))  # 扩展列表，确保足够数量的样本
                self.ids = self.ids[:nsample]  # 截取指定数量的样本
        else:  # 验证模式
            with open("splits/%s/val.txt" % name, "r") as f:  # 从验证分割文件读取ID
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        """
        根据数据集中的索引 (item) 检索图像及其相应的掩码。
        图像被打开并转换为RGB；掩码被加载并转换为数组，然后返回图像以实现处理兼容性。
        """
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(" ")[0])).convert("RGB")
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(" ")[1]))))
        """
        如果是验证模式，则对图像及掩码进行归一化处理后返回；
        """
        if self.mode == "val":
            img, mask = normalize(img, mask)  # 应用归一化
            return img, mask, id
        """对输入图像应用裁剪和翻转等弱扰动"""
        img, mask = resize(img, mask, (0.5, 2.0))  # 调整大小
        ignore_value = 254 if self.mode == "train_u" else 255  # 设置忽略值
        img, mask = crop(img, mask, self.size, ignore_value)  # 裁剪
        img, mask = hflip(img, mask, p=0.5)  # 水平翻转
        """
        如果是有标签的训练模式，则对图像及掩码进行归一化处理后返回；
        """
        if self.mode == "train_l":
            return normalize(img, mask)

        """
        对于无标签训练模式，图像会经历多次变换：颜色抖动、随机灰度、模糊、CutMix等；
        """
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)  # 创建图片副本
        if random.random() < 0.8:  # 随机应用颜色抖动
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)  # 应用随机灰度变换
        img_s1 = blur(img_s1, p=0.5)  # 应用模糊
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)  # 获取CutMix盒子

        if random.random() < 0.8:  # 同上，对第二个副本进行处理
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))  # 创建忽略掩码

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)  # 归一化第一个副本和忽略掩码
        img_s2 = normalize(img_s2)  # 归一化第二个副本

        mask = torch.from_numpy(np.array(mask)).long()  # 将掩码转换为张量
        ignore_mask[mask == 254] = 255  # 设置忽略掩码的忽略值

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2  # 返回处理后的数据集

    def __len__(self):
        return len(self.ids)
