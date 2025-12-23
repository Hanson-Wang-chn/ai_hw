# -*- coding: utf-8 -*-
"""
RNN Seq2Seq模型实现
包含LSTM编码器、Bahdanau注意力机制和解码器
用于与Transformer进行基础对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class BahdanauAttention(nn.Module):
    """
    Bahdanau注意力机制（加性注意力）
    score(s_t, h_i) = v^T * tanh(W_s * s_t + W_h * h_i)
    """

    def __init__(self, hidden_size: int, attention_size: int = 256):
        super().__init__()
        self.W_s = nn.Linear(hidden_size, attention_size, bias=False)
        self.W_h = nn.Linear(hidden_size * 2, attention_size, bias=False)  # 双向LSTM
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: 解码器隐藏状态 (batch_size, hidden_size)
            encoder_outputs: 编码器输出 (batch_size, src_len, hidden_size * 2)
            mask: 编码器掩码 (batch_size, src_len)

        Returns:
            context: 上下文向量 (batch_size, hidden_size * 2)
            attention_weights: 注意力权重 (batch_size, src_len)
        """
        batch_size, src_len, _ = encoder_outputs.size()

        # 扩展解码器隐藏状态
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # 计算注意力分数
        energy = torch.tanh(self.W_s(decoder_hidden) + self.W_h(encoder_outputs))
        attention_scores = self.v(energy).squeeze(-1)  # (batch_size, src_len)

        # 应用掩码
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class RNNEncoder(nn.Module):
    """双向LSTM编码器"""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

        # 用于将双向LSTM的隐藏状态转换为解码器初始状态
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: 源序列 (batch_size, src_len)
            src_lengths: 源序列长度（用于pack_padded_sequence）

        Returns:
            encoder_outputs: (batch_size, src_len, hidden_size * 2)
            (hidden, cell): 解码器初始状态
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(src))

        # LSTM编码
        if src_lengths is not None:
            # 使用pack_padded_sequence处理变长序列
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs, batch_first=True
            )
        else:
            encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # 合并双向隐藏状态
        # hidden: (num_layers * 2, batch_size, hidden_size)
        # 需要转换为: (num_layers, batch_size, hidden_size)
        hidden = self._combine_directions(hidden)
        cell = self._combine_directions(cell)

        # 线性变换得到解码器初始状态
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))

        return encoder_outputs, (hidden, cell)

    def _combine_directions(self, states: torch.Tensor) -> torch.Tensor:
        """合并双向LSTM的隐藏状态"""
        # states: (num_layers * 2, batch_size, hidden_size)
        num_layers = states.size(0) // 2
        batch_size = states.size(1)
        hidden_size = states.size(2)

        # 重塑为 (num_layers, 2, batch_size, hidden_size)
        states = states.view(num_layers, 2, batch_size, hidden_size)

        # 拼接前向和后向: (num_layers, batch_size, hidden_size * 2)
        states = torch.cat([states[:, 0, :, :], states[:, 1, :, :]], dim=-1)

        return states


class RNNDecoder(nn.Module):
    """带注意力机制的LSTM解码器"""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_size: int = 256
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(hidden_size, attention_size)

        # 输入为词嵌入 + 上下文向量
        self.lstm = nn.LSTM(
            embed_size + hidden_size * 2,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc_out = nn.Linear(hidden_size * 3 + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步解码

        Args:
            input_token: 输入token (batch_size,)
            hidden: LSTM隐藏状态 (num_layers, batch_size, hidden_size)
            cell: LSTM细胞状态 (num_layers, batch_size, hidden_size)
            encoder_outputs: 编码器输出 (batch_size, src_len, hidden_size * 2)
            mask: 编码器掩码

        Returns:
            output: 输出logits (batch_size, vocab_size)
            hidden: 更新后的隐藏状态
            cell: 更新后的细胞状态
            attention_weights: 注意力权重
        """
        # 词嵌入 (batch_size, 1, embed_size)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))

        # 计算注意力
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, mask
        )

        # 拼接嵌入和上下文向量
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)

        # LSTM解码
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # 计算输出
        output = torch.cat([
            lstm_output.squeeze(1),
            context,
            embedded.squeeze(1)
        ], dim=-1)
        output = self.fc_out(output)

        return output, hidden, cell, attention_weights


class RNNSeq2Seq(nn.Module):
    """完整的RNN Seq2Seq模型（带Bahdanau注意力）"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_size: int = 256,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = RNNEncoder(
            src_vocab_size, embed_size, hidden_size, num_layers, dropout
        )
        self.decoder = RNNDecoder(
            tgt_vocab_size, embed_size, hidden_size, num_layers, dropout, attention_size
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            src: 源序列 (batch_size, src_len)
            tgt: 目标序列 (batch_size, tgt_len)
            teacher_forcing_ratio: 教师强制比例
            return_attention: 是否返回注意力权重

        Returns:
            包含输出和可选注意力权重的字典
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # 创建掩码
        src_mask = (src != self.pad_idx)

        # 编码
        encoder_outputs, (hidden, cell) = self.encoder(src)

        # 初始化输出
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)
        all_attention_weights = []

        # 解码器输入从<sos>开始
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden, cell, attention_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            outputs[:, t, :] = output

            if return_attention:
                all_attention_weights.append(attention_weights)

            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = tgt[:, t] if teacher_force else top1

        result = {'output': outputs, 'encoder_output': encoder_outputs}

        if return_attention:
            result['attention_weights'] = torch.stack(all_attention_weights, dim=1)

        return result

    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_rnn_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config: Dict,
    pad_idx: int = 0
) -> RNNSeq2Seq:
    """
    创建RNN模型的工厂函数

    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        config: 模型配置字典
        pad_idx: 填充token索引

    Returns:
        RNNSeq2Seq模型
    """
    model = RNNSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_size=config.get('embed_size', 512),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        attention_size=config.get('attention_size', 256),
        pad_idx=pad_idx
    )

    print(f"RNN模型参数量: {model.count_parameters():,}")
    return model
