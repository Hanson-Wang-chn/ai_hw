# -*- coding: utf-8 -*-
"""
位置编码模块
实现不同类型的位置编码：正弦位置编码、可学习位置编码、相对位置编码
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦位置编码 (Vaswani et al., 2017)
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

    def get_pe_matrix(self, seq_len: int) -> torch.Tensor:
        """获取位置编码矩阵，用于可视化"""
        return self.pe[:, :seq_len, :].squeeze(0)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码
    位置编码作为可训练参数，随模型训练更新
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 可学习的位置嵌入
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

    def get_pe_matrix(self, seq_len: int) -> torch.Tensor:
        """获取位置编码矩阵，用于可视化"""
        return self.pe[:, :seq_len, :].squeeze(0).detach()


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码 (Shaw et al., 2018)
    编码token之间的相对距离而非绝对位置
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # 相对位置嵌入：考虑正负相对距离
        self.relative_positions = 2 * max_len - 1
        self.pe = nn.Embedding(self.relative_positions, d_model)

        # 初始化
        nn.init.xavier_uniform_(self.pe.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            添加位置编码后的张量（相对位置编码在注意力计算时使用）
        """
        # 对于基础的序列编码，返回原始输入
        # 相对位置编码主要在注意力计算中使用
        return self.dropout(x)

    def get_relative_positions_matrix(self, seq_len: int) -> torch.Tensor:
        """
        获取相对位置矩阵

        Args:
            seq_len: 序列长度

        Returns:
            相对位置矩阵，形状为 (seq_len, seq_len)
        """
        # 计算相对位置索引
        range_vec = torch.arange(seq_len)
        distance_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        # 将负数偏移到正数范围
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_len + 1, self.max_len - 1)
        final_mat = distance_mat_clipped + self.max_len - 1
        return final_mat

    def get_relative_embeddings(self, seq_len: int) -> torch.Tensor:
        """
        获取相对位置嵌入

        Args:
            seq_len: 序列长度

        Returns:
            相对位置嵌入，形状为 (seq_len, seq_len, d_model)
        """
        relative_positions = self.get_relative_positions_matrix(seq_len)
        relative_positions = relative_positions.to(self.pe.weight.device)
        return self.pe(relative_positions)

    def get_pe_matrix(self, seq_len: int) -> torch.Tensor:
        """获取位置编码矩阵，用于可视化（返回相对位置嵌入的平均）"""
        rel_emb = self.get_relative_embeddings(seq_len)
        # 返回对角线附近的嵌入作为可视化
        return rel_emb.mean(dim=1).detach()


def get_positional_encoding(pe_type: str, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> nn.Module:
    """
    获取位置编码模块的工厂函数

    Args:
        pe_type: 位置编码类型
            - 'sinusoidal': 正弦位置编码
            - 'learnable': 可学习位置编码
            - 'relative': 相对位置编码
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout概率

    Returns:
        位置编码模块
    """
    if pe_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)
    elif pe_type == 'learnable':
        return LearnablePositionalEncoding(d_model, max_len, dropout)
    elif pe_type == 'relative':
        return RelativePositionalEncoding(d_model, max_len, dropout)
    else:
        raise ValueError(f"未知的位置编码类型: {pe_type}")


def compute_pe_similarity(pe_module: nn.Module, seq_len: int = 20) -> torch.Tensor:
    """
    计算位置编码向量之间的相似度矩阵

    Args:
        pe_module: 位置编码模块
        seq_len: 序列长度

    Returns:
        相似度矩阵，形状为 (seq_len, seq_len)
    """
    pe_matrix = pe_module.get_pe_matrix(seq_len)

    # 归一化
    pe_norm = pe_matrix / (pe_matrix.norm(dim=-1, keepdim=True) + 1e-8)

    # 计算余弦相似度
    similarity = torch.mm(pe_norm, pe_norm.t())

    return similarity
