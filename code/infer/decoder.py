# -*- coding: utf-8 -*-
"""
解码策略实现
包含贪心搜索、束搜索、温度采样等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    sos_idx: int,
    eos_idx: int,
    max_len: int = 50,
    device: torch.device = None
) -> torch.Tensor:
    """
    贪心解码

    Args:
        model: Seq2Seq模型
        src: 源序列 (batch_size, src_len)
        sos_idx: 起始token索引
        eos_idx: 结束token索引
        max_len: 最大生成长度
        device: 计算设备

    Returns:
        生成的序列 (batch_size, gen_len)
    """
    if device is None:
        device = src.device

    model.eval()
    batch_size = src.size(0)

    # 编码
    with torch.no_grad():
        # 创建源序列掩码
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        encoder_output, _ = model.encoder(src, src_mask)

    # 初始化目标序列
    tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            # 创建目标掩码
            tgt_len = tgt.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            # 解码
            decoder_output, _, _ = model.decoder(
                tgt, encoder_output, tgt_mask, src_mask
            )

        # 获取最后一个时间步的预测
        next_token_logits = decoder_output[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        # 更新目标序列
        tgt = torch.cat([tgt, next_token], dim=1)

        # 检查是否完成
        finished = finished | (next_token.squeeze(-1) == eos_idx)
        if finished.all():
            break

    return tgt


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    sos_idx: int,
    eos_idx: int,
    beam_width: int = 4,
    max_len: int = 50,
    length_penalty: float = 0.6,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    束搜索解码

    Args:
        model: Seq2Seq模型
        src: 源序列 (batch_size, src_len)
        sos_idx: 起始token索引
        eos_idx: 结束token索引
        beam_width: 束宽
        max_len: 最大生成长度
        length_penalty: 长度惩罚系数
        device: 计算设备

    Returns:
        best_sequences: 最佳序列 (batch_size, gen_len)
        best_scores: 最佳分数 (batch_size,)
    """
    if device is None:
        device = src.device

    model.eval()
    batch_size = src.size(0)

    # 编码
    with torch.no_grad():
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        encoder_output, _ = model.encoder(src, src_mask)

    # 扩展batch以适应beam search
    encoder_output = encoder_output.unsqueeze(1).repeat(1, beam_width, 1, 1)
    encoder_output = encoder_output.view(batch_size * beam_width, -1, encoder_output.size(-1))
    src_mask = src_mask.unsqueeze(1).repeat(1, beam_width, 1, 1, 1)
    src_mask = src_mask.view(batch_size * beam_width, 1, 1, -1)

    # 初始化beam
    beam_scores = torch.zeros(batch_size, beam_width, device=device)
    beam_scores[:, 1:] = float('-inf')  # 初始时只有第一个beam有效

    # 初始化序列
    sequences = torch.full(
        (batch_size * beam_width, 1), sos_idx,
        dtype=torch.long, device=device
    )

    finished = torch.zeros(batch_size * beam_width, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        with torch.no_grad():
            # 创建目标掩码
            tgt_len = sequences.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            # 解码
            decoder_output, _, _ = model.decoder(
                sequences, encoder_output, tgt_mask, src_mask
            )

        # 获取下一个token的概率
        next_token_logits = decoder_output[:, -1, :]
        next_token_probs = F.log_softmax(next_token_logits, dim=-1)

        vocab_size = next_token_probs.size(-1)

        # 重塑为 (batch_size, beam_width, vocab_size)
        next_token_probs = next_token_probs.view(batch_size, beam_width, vocab_size)

        # 计算累积分数
        next_scores = beam_scores.unsqueeze(-1) + next_token_probs

        # 展平并选择top-k
        next_scores = next_scores.view(batch_size, -1)
        top_scores, top_indices = next_scores.topk(beam_width, dim=-1)

        # 计算beam索引和token索引
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        # 更新序列
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        prev_sequences = sequences.view(batch_size, beam_width, -1)
        selected_sequences = prev_sequences[batch_indices, beam_indices]
        sequences = torch.cat([
            selected_sequences,
            token_indices.unsqueeze(-1)
        ], dim=-1)
        sequences = sequences.view(batch_size * beam_width, -1)

        # 更新分数
        beam_scores = top_scores

        # 应用长度惩罚
        length = step + 2
        normalized_scores = beam_scores / (length ** length_penalty)

        # 检查是否完成
        finished = (token_indices == eos_idx).view(-1)

    # 选择最佳序列
    best_indices = beam_scores.argmax(dim=-1)
    best_sequences = sequences.view(batch_size, beam_width, -1)
    batch_indices = torch.arange(batch_size, device=device)
    best_sequences = best_sequences[batch_indices, best_indices]

    best_scores = beam_scores[batch_indices, best_indices]

    return best_sequences, best_scores


def temperature_sampling_decode(
    model: nn.Module,
    src: torch.Tensor,
    sos_idx: int,
    eos_idx: int,
    temperature: float = 1.0,
    max_len: int = 50,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    温度采样解码

    Args:
        model: Seq2Seq模型
        src: 源序列 (batch_size, src_len)
        sos_idx: 起始token索引
        eos_idx: 结束token索引
        temperature: 温度参数（越高越随机）
        max_len: 最大生成长度
        top_k: Top-K采样参数（0表示不使用）
        top_p: Top-P (nucleus) 采样参数
        device: 计算设备

    Returns:
        sequences: 生成的序列 (batch_size, gen_len)
        token_probs: 每个token的选择概率 (batch_size, gen_len)
    """
    if device is None:
        device = src.device

    model.eval()
    batch_size = src.size(0)

    # 编码
    with torch.no_grad():
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        encoder_output, _ = model.encoder(src, src_mask)

    # 初始化目标序列
    tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    all_probs = []

    for _ in range(max_len - 1):
        with torch.no_grad():
            # 创建目标掩码
            tgt_len = tgt.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            # 解码
            decoder_output, _, _ = model.decoder(
                tgt, encoder_output, tgt_mask, src_mask
            )

        # 获取最后一个时间步的预测
        next_token_logits = decoder_output[:, -1, :] / temperature

        # Top-K采样
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Top-P (nucleus) 采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过阈值的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')

        # 采样
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 记录概率
        token_prob = probs.gather(1, next_token)
        all_probs.append(token_prob)

        # 更新目标序列
        tgt = torch.cat([tgt, next_token], dim=1)

        # 检查是否完成
        finished = finished | (next_token.squeeze(-1) == eos_idx)
        if finished.all():
            break

    token_probs = torch.cat(all_probs, dim=1) if all_probs else torch.zeros(batch_size, 0, device=device)

    return tgt, token_probs


def get_decoding_probability_distribution(
    model: nn.Module,
    src: torch.Tensor,
    tgt_prefix: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """
    获取解码概率分布（用于可视化）

    Args:
        model: Seq2Seq模型
        src: 源序列 (batch_size, src_len)
        tgt_prefix: 目标序列前缀 (batch_size, prefix_len)
        device: 计算设备

    Returns:
        概率分布 (batch_size, vocab_size)
    """
    if device is None:
        device = src.device

    model.eval()

    with torch.no_grad():
        # 编码
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        encoder_output, _ = model.encoder(src, src_mask)

        # 创建目标掩码
        tgt_len = tgt_prefix.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        # 解码
        decoder_output, _, _ = model.decoder(
            tgt_prefix, encoder_output, tgt_mask, src_mask
        )

    # 获取最后一个时间步的概率分布
    logits = decoder_output[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    return probs


class Decoder:
    """解码器统一接口"""

    def __init__(
        self,
        model: nn.Module,
        sos_idx: int,
        eos_idx: int,
        device: torch.device
    ):
        self.model = model
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def decode(
        self,
        src: torch.Tensor,
        strategy: str = 'greedy',
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        解码

        Args:
            src: 源序列
            strategy: 解码策略
                - 'greedy': 贪心搜索
                - 'beam': 束搜索
                - 'sampling': 温度采样
            **kwargs: 策略特定参数

        Returns:
            包含序列和相关信息的字典
        """
        if strategy == 'greedy':
            sequences = greedy_decode(
                self.model, src, self.sos_idx, self.eos_idx,
                max_len=kwargs.get('max_len', 50),
                device=self.device
            )
            return {'sequences': sequences}

        elif strategy == 'beam':
            sequences, scores = beam_search_decode(
                self.model, src, self.sos_idx, self.eos_idx,
                beam_width=kwargs.get('beam_width', 4),
                max_len=kwargs.get('max_len', 50),
                length_penalty=kwargs.get('length_penalty', 0.6),
                device=self.device
            )
            return {'sequences': sequences, 'scores': scores}

        elif strategy == 'sampling':
            sequences, probs = temperature_sampling_decode(
                self.model, src, self.sos_idx, self.eos_idx,
                temperature=kwargs.get('temperature', 1.0),
                max_len=kwargs.get('max_len', 50),
                top_k=kwargs.get('top_k', 0),
                top_p=kwargs.get('top_p', 1.0),
                device=self.device
            )
            return {'sequences': sequences, 'token_probs': probs}

        else:
            raise ValueError(f"未知的解码策略: {strategy}")
