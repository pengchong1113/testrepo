# @file
# @brief    TID detecting model training
# @details  TID detecting model training
# @founder  Xiaodong Ren, E-mail: xdren@whu.edu.cn
#           PLANET PPP - ION Group, SGG, WHU
# @author   Pengchong Zhao
# @date     2024/04/23
# @version  1.0.0
# @par      Copyright(c) 2024-2099 School of Geodesy and Geomatics, University of Wuhan. All Rights Reserved.
# @par      History:
#           2024/04/23 Pengchong Zhao 首次创建该文件，后续如有功能变更，请参考本条记录方式增加说明
import os
import sys

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录、以及transforms模块所在的目录添加到Python路径中
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))
# 调用TID_detec_model_train.py中的main函数
from src.TID_detec_model_train import TID_model_train

def train():
    # 定义参数
    class Args:
        def __init__(self):
            self.device = 'cuda:0'  # 或者 'cpu'，选择设备
            self.data_path = './data/VOCdevkit'  # 数据集根目录
            self.num_classes = 2  # 目标类别数（不包含背景）
            self.output_dir = './save_weights/'  # 权重保存地址
            self.resume = ''  # 恢复训练的权重文件地址
            self.start_epoch = 0  # 开始训练的epoch数
            self.epochs = 30  # 训练总epoch数
            self.lr = 0.01  # 学习率
            self.momentum = 0.9  # SGD的momentum参数
            self.weight_decay = 1e-4  # SGD的weight_decay参数
            self.lr_steps = [10, 20]  # torch.optim.lr_scheduler.MultiStepLR的参数
            self.lr_gamma = 0.1  # torch.optim.lr_scheduler.MultiStepLR的参数
            self.batch_size = 2  # 训练的batch size
            self.aspect_ratio_group_factor = 3  # 按图片相似高宽比采样图片组成batch的参数
            self.pretrain = True  # 是否加载COCO预训练权重
            self.amp = False  # 是否使用混合精度训练

    args = Args()

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 调用main函数开始训练
    TID_model_train(args)

if __name__ == "__main__":
    train()
