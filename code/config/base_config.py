# -*- coding: utf-8 -*-
"""
基准超参数配置
包含所有实验共享的固定超参数
"""

import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据配置
DATA_CONFIG = {
    'max_seq_len': 20,          # 序列最大长度
    'min_freq': 2,              # 词汇表最小词频
    'batch_size': 32,           # 批次大小
    'max_token_len': 30,        # 过滤长度超过30的样本
}

# 特殊token
SPECIAL_TOKENS = {
    'pad_token': '<pad>',
    'sos_token': '<sos>',
    'eos_token': '<eos>',
    'unk_token': '<unk>',
}

# 模型基础配置
MODEL_CONFIG = {
    'd_model': 512,             # 模型维度
    'nhead': 8,                 # 注意力头数
    'num_encoder_layers': 3,    # 编码器层数
    'num_decoder_layers': 3,    # 解码器层数
    'dim_feedforward': 2048,    # FFN隐藏层维度
    'dropout': 0.1,             # Dropout概率
    'activation': 'relu',       # 激活函数
}

# 训练配置
TRAIN_CONFIG = {
    'lr': 1e-4,                 # 学习率
    'weight_decay': 0.01,       # 权重衰减
    'betas': (0.9, 0.98),       # Adam优化器参数
    'epochs': 50,               # 训练轮次
    'patience': 5,              # 早停patience
    'lr_patience': 3,           # 学习率调度patience
    'lr_factor': 0.5,           # 学习率衰减因子
    'grad_clip': 1.0,           # 梯度裁剪
    'label_smoothing': 0.1,     # 标签平滑
}

# 评估配置
EVAL_CONFIG = {
    'beam_width': 4,            # 束搜索宽度
    'temperature': 1.0,         # 采样温度
}

# 日志配置
LOG_CONFIG = {
    'log_interval': 100,        # 日志打印间隔
    'save_best_only': True,     # 仅保存最优模型
}

# RNN模型配置（用于对比实验）
RNN_CONFIG = {
    'hidden_size': 512,         # 隐藏层大小
    'num_layers': 2,            # 层数
    'dropout': 0.3,             # Dropout概率
    'bidirectional': True,      # 是否双向
}
