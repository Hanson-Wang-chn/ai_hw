# -*- coding: utf-8 -*-
"""
注意力机制模块
实现不同类型的注意力机制：标准点积注意力、多头注意力、线性注意力、双向注意力
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    标准缩放点积注意力 (Vaswani et al., 2017)
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: 注意力掩码
            return_attention: 是否返回注意力权重

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: 可选的注意力权重
        """
        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)
        output = self.W_o(output)

        if return_attention:
            return output, attention_weights
        return output, None


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    允许模型同时关注不同位置的不同表示子空间
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: 注意力掩码
            return_attention: 是否返回注意力权重

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: 可选的注意力权重 (batch_size, nhead, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 线性变换并分割为多头
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        if return_attention:
            return output, attention_weights
        return output, None


class LinearAttention(nn.Module):
    """
    线性注意力机制 (Choromanski et al., 2020)
    使用核函数近似softmax，将复杂度从O(n^2)降低到O(n)
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.eps = eps

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """使用ELU+1作为特征映射函数"""
        return F.elu(x) + 1

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: 注意力掩码（线性注意力中的处理方式不同）
            return_attention: 是否返回注意力权重

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: 可选的注意力权重
        """
        batch_size = query.size(0)

        # 线性变换并分割为多头
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 应用特征映射
        Q = self._feature_map(Q)
        K = self._feature_map(K)

        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(-1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            K = K * mask.float()
            V = V * mask.float()

        # 线性注意力计算: (Q @ (K^T @ V)) / (Q @ sum(K))
        KV = torch.matmul(K.transpose(-2, -1), V)  # (batch, nhead, d_k, d_k)
        output = torch.matmul(Q, KV)  # (batch, nhead, seq_len, d_k)

        # 归一化
        K_sum = K.sum(dim=-2, keepdim=True)  # (batch, nhead, 1, d_k)
        normalizer = torch.matmul(Q, K_sum.transpose(-2, -1)) + self.eps
        output = output / normalizer

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)

        # 线性注意力无法高效返回完整注意力矩阵
        return output, None


class BidirectionalAttention(nn.Module):
    """
    双向注意力机制
    Encoder端采用双向自注意力，Decoder端采用单向自注意力+交叉注意力
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 双向自注意力（用于Encoder）
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        双向注意力前向传播
        """
        return self.self_attn(query, key, value, mask, return_attention)


def get_attention_module(
    attn_type: str,
    d_model: int,
    nhead: int = 8,
    dropout: float = 0.1
) -> nn.Module:
    """
    获取注意力模块的工厂函数

    Args:
        attn_type: 注意力类型
            - 'scaled_dot': 标准缩放点积注意力
            - 'multi_head': 多头注意力
            - 'linear': 线性注意力
            - 'bidirectional': 双向注意力
        d_model: 模型维度
        nhead: 注意力头数
        dropout: Dropout概率

    Returns:
        注意力模块
    """
    if attn_type == 'scaled_dot':
        return ScaledDotProductAttention(d_model, dropout)
    elif attn_type == 'multi_head':
        return MultiHeadAttention(d_model, nhead, dropout)
    elif attn_type == 'linear':
        return LinearAttention(d_model, nhead, dropout)
    elif attn_type == 'bidirectional':
        return BidirectionalAttention(d_model, nhead, dropout)
    else:
        raise ValueError(f"未知的注意力类型: {attn_type}")


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    计算注意力权重分布的熵
    熵值越低表示注意力越集中

    Args:
        attention_weights: 注意力权重 (batch_size, nhead, seq_len_q, seq_len_k)
                          或 (batch_size, seq_len_q, seq_len_k)

    Returns:
        注意力熵 (batch_size,) 或标量
    """
    # 避免log(0)
    eps = 1e-10
    attn = attention_weights + eps

    # 计算熵: -sum(p * log(p))
    entropy = -torch.sum(attn * torch.log(attn), dim=-1)

    # 返回平均熵
    return entropy.mean()
