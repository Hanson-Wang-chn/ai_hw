# -*- coding: utf-8 -*-
"""
基准Transformer实现
包含Encoder、Decoder和完整的Seq2Seq模型
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from model.pe_module import get_positional_encoding
from model.attention_module import MultiHeadAttention, get_attention_module


class FeedForward(nn.Module):
    """前馈神经网络模块"""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        # 选择激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"未知的激活函数: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_type: str = 'post',  # 'pre' or 'post'
        attn_type: str = 'multi_head'
    ):
        super().__init__()
        self.norm_type = norm_type

        # 自注意力层
        self.self_attn = get_attention_module(attn_type, d_model, nhead, dropout)

        # 前馈网络
        self.ffn = FeedForward(d_model, dim_feedforward, dropout, activation)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src: (batch_size, seq_len, d_model)
            src_mask: 源序列掩码
            return_attention: 是否返回注意力权重

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: 可选的注意力权重
        """
        if self.norm_type == 'pre':
            # Pre-LN: LayerNorm在注意力/FFN之前
            src_norm = self.norm1(src)
            attn_output, attn_weights = self.self_attn(
                src_norm, src_norm, src_norm, src_mask, return_attention
            )
            src = src + self.dropout(attn_output)

            src_norm = self.norm2(src)
            ffn_output = self.ffn(src_norm)
            src = src + ffn_output
        else:
            # Post-LN: LayerNorm在注意力/FFN之后
            attn_output, attn_weights = self.self_attn(
                src, src, src, src_mask, return_attention
            )
            src = self.norm1(src + self.dropout(attn_output))
            ffn_output = self.ffn(src)
            src = self.norm2(src + ffn_output)

        if return_attention:
            return src, attn_weights
        return src, None


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_type: str = 'post',
        attn_type: str = 'multi_head'
    ):
        super().__init__()
        self.norm_type = norm_type

        # 自注意力层（带因果掩码）
        self.self_attn = get_attention_module(attn_type, d_model, nhead, dropout)

        # 交叉注意力层
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)

        # 前馈网络
        self.ffn = FeedForward(d_model, dim_feedforward, dropout, activation)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            tgt: (batch_size, tgt_len, d_model)
            memory: 编码器输出 (batch_size, src_len, d_model)
            tgt_mask: 目标序列掩码（因果掩码）
            memory_mask: 编码器输出掩码
            return_attention: 是否返回注意力权重

        Returns:
            output: (batch_size, tgt_len, d_model)
            self_attn_weights: 自注意力权重
            cross_attn_weights: 交叉注意力权重
        """
        if self.norm_type == 'pre':
            # Pre-LN
            tgt_norm = self.norm1(tgt)
            self_attn_output, self_attn_weights = self.self_attn(
                tgt_norm, tgt_norm, tgt_norm, tgt_mask, return_attention
            )
            tgt = tgt + self.dropout(self_attn_output)

            tgt_norm = self.norm2(tgt)
            cross_attn_output, cross_attn_weights = self.cross_attn(
                tgt_norm, memory, memory, memory_mask, return_attention
            )
            tgt = tgt + self.dropout(cross_attn_output)

            tgt_norm = self.norm3(tgt)
            ffn_output = self.ffn(tgt_norm)
            tgt = tgt + ffn_output
        else:
            # Post-LN
            self_attn_output, self_attn_weights = self.self_attn(
                tgt, tgt, tgt, tgt_mask, return_attention
            )
            tgt = self.norm1(tgt + self.dropout(self_attn_output))

            cross_attn_output, cross_attn_weights = self.cross_attn(
                tgt, memory, memory, memory_mask, return_attention
            )
            tgt = self.norm2(tgt + self.dropout(cross_attn_output))

            ffn_output = self.ffn(tgt)
            tgt = self.norm3(tgt + ffn_output)

        if return_attention:
            return tgt, self_attn_weights, cross_attn_weights
        return tgt, None, None


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        pe_type: str = 'sinusoidal',
        norm_type: str = 'post',
        attn_type: str = 'multi_head',
        max_len: int = 5000
    ):
        super().__init__()
        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoder = get_positional_encoding(pe_type, d_model, max_len, dropout)

        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, norm_type, attn_type
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            src: 源序列索引 (batch_size, src_len)
            src_mask: 源序列掩码
            return_attention: 是否返回注意力权重

        Returns:
            output: (batch_size, src_len, d_model)
            all_attention_weights: 所有层的注意力权重列表
        """
        # 词嵌入
        x = self.embedding(src) * math.sqrt(self.d_model)

        # 位置编码
        x = self.pos_encoder(x)

        all_attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask, return_attention)
            if return_attention:
                all_attention_weights.append(attn_weights)

        x = self.norm(x)

        if return_attention:
            return x, all_attention_weights
        return x, None


class TransformerDecoder(nn.Module):
    """Transformer解码器"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        pe_type: str = 'sinusoidal',
        norm_type: str = 'post',
        attn_type: str = 'multi_head',
        max_len: int = 5000
    ):
        super().__init__()
        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_decoder = get_positional_encoding(pe_type, d_model, max_len, dropout)

        # 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, norm_type, attn_type
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 输出投影
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list], Optional[list]]:
        """
        Args:
            tgt: 目标序列索引 (batch_size, tgt_len)
            memory: 编码器输出 (batch_size, src_len, d_model)
            tgt_mask: 目标序列掩码（因果掩码）
            memory_mask: 编码器输出掩码
            return_attention: 是否返回注意力权重

        Returns:
            output: (batch_size, tgt_len, vocab_size)
            all_self_attn_weights: 所有层的自注意力权重列表
            all_cross_attn_weights: 所有层的交叉注意力权重列表
        """
        # 词嵌入
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        # 位置编码
        x = self.pos_decoder(x)

        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, memory, tgt_mask, memory_mask, return_attention)
            if return_attention:
                all_self_attn_weights.append(self_attn)
                all_cross_attn_weights.append(cross_attn)

        x = self.norm(x)
        output = self.output_projection(x)

        if return_attention:
            return output, all_self_attn_weights, all_cross_attn_weights
        return output, None, None


class TransformerSeq2Seq(nn.Module):
    """完整的Transformer Seq2Seq模型"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        pe_type: str = 'sinusoidal',
        norm_type: str = 'post',
        attn_type: str = 'multi_head',
        max_len: int = 5000,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # 编码器
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, nhead, num_encoder_layers,
            dim_feedforward, dropout, activation, pe_type, norm_type, attn_type, max_len
        )

        # 解码器
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, nhead, num_decoder_layers,
            dim_feedforward, dropout, activation, pe_type, norm_type, attn_type, max_len
        )

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码（上三角为-inf）"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def create_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建注意力掩码

        Args:
            src: 源序列 (batch_size, src_len)
            tgt: 目标序列 (batch_size, tgt_len)

        Returns:
            src_mask: 源序列填充掩码
            tgt_mask: 目标序列因果掩码
            memory_mask: 编码器输出掩码
        """
        # 源序列填充掩码
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # 目标序列因果掩码
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
        ).bool()
        tgt_causal_mask = ~tgt_causal_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & tgt_causal_mask

        # 编码器输出掩码（与源序列填充掩码相同）
        memory_mask = src_mask

        return src_mask, tgt_mask, memory_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            src: 源序列 (batch_size, src_len)
            tgt: 目标序列 (batch_size, tgt_len)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            memory_mask: 编码器输出掩码
            return_attention: 是否返回注意力权重

        Returns:
            包含输出和可选注意力权重的字典
        """
        # 创建掩码
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask, memory_mask = self.create_masks(src, tgt)

        # 编码
        encoder_output, enc_attn = self.encoder(src, src_mask, return_attention)

        # 解码
        decoder_output, dec_self_attn, dec_cross_attn = self.decoder(
            tgt, encoder_output, tgt_mask, memory_mask, return_attention
        )

        result = {'output': decoder_output, 'encoder_output': encoder_output}

        if return_attention:
            result['encoder_attention'] = enc_attn
            result['decoder_self_attention'] = dec_self_attn
            result['decoder_cross_attention'] = dec_cross_attn

        return result

    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config: Dict,
    pad_idx: int = 0
) -> TransformerSeq2Seq:
    """
    创建Transformer模型的工厂函数

    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        config: 模型配置字典
        pad_idx: 填充token索引

    Returns:
        TransformerSeq2Seq模型
    """
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 3),
        num_decoder_layers=config.get('num_decoder_layers', 3),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        pe_type=config.get('pe_type', 'sinusoidal'),
        norm_type=config.get('norm_type', 'post'),
        attn_type=config.get('attn_type', 'multi_head'),
        max_len=config.get('max_len', 5000),
        pad_idx=pad_idx
    )

    print(f"模型参数量: {model.count_parameters():,}")
    return model
