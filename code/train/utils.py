# -*- coding: utf-8 -*-
"""
训练工具函数
包含损失计算、评估指标、日志记录等
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def setup_logger(log_dir: str, name: str = 'train') -> logging.Logger:
    """
    设置日志记录器

    Args:
        log_dir: 日志目录
        name: 日志名称

    Returns:
        配置好的日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除已有的处理器
    logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""

    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (batch_size * seq_len, vocab_size)
            target: 目标索引 (batch_size * seq_len,)

        Returns:
            平滑后的交叉熵损失
        """
        # 创建平滑标签
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # 排除pad和真实标签
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0

        # 创建掩码
        mask = (target != self.pad_idx)
        true_dist = true_dist * mask.unsqueeze(1)

        # 计算KL散度
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(true_dist * log_probs) / mask.sum()

        return loss


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
    smoothing: float = 0.0
) -> torch.Tensor:
    """
    计算损失

    Args:
        pred: 预测logits (batch_size, seq_len, vocab_size)
        target: 目标索引 (batch_size, seq_len)
        pad_idx: 填充token索引
        smoothing: 标签平滑系数

    Returns:
        损失值
    """
    # 移除第一个时间步（<sos>）
    pred = pred[:, 1:, :].contiguous()
    target = target[:, 1:].contiguous()

    # 展平
    pred = pred.view(-1, pred.size(-1))
    target = target.view(-1)

    if smoothing > 0:
        loss_fn = LabelSmoothingLoss(pred.size(-1), smoothing, pad_idx)
        loss = loss_fn(pred, target)
    else:
        loss = F.cross_entropy(pred, target, ignore_index=pad_idx)

    return loss


def compute_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    n_gram: int = 4
) -> Dict[str, float]:
    """
    计算BLEU分数

    Args:
        hypotheses: 预测文本列表
        references: 参考文本列表
        n_gram: 最大n-gram

    Returns:
        包含BLEU-1/2/4的字典
    """
    smoothie = SmoothingFunction().method1

    bleu_scores = {f'bleu_{i}': [] for i in [1, 2, 4]}

    for hyp, ref in zip(hypotheses, references):
        # 计算不同n-gram的BLEU
        for n in [1, 2, 4]:
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            score = sentence_bleu(
                [ref], hyp,
                weights=weights,
                smoothing_function=smoothie
            )
            bleu_scores[f'bleu_{n}'].append(score)

    # 计算平均分数
    return {k: np.mean(v) for k, v in bleu_scores.items()}


def compute_rouge(
    hypotheses: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    计算ROUGE-L分数

    Args:
        hypotheses: 预测文本列表
        references: 参考文本列表

    Returns:
        包含ROUGE-L的字典
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_scores = []
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        rouge_scores.append(scores['rougeL'].fmeasure)

    return {'rouge_l': np.mean(rouge_scores)}


def compute_bert_score(
    hypotheses: List[str],
    references: List[str],
    lang: str = 'de'
) -> Dict[str, float]:
    """
    计算BERTScore

    Args:
        hypotheses: 预测文本列表
        references: 参考文本列表
        lang: 语言

    Returns:
        包含BERTScore的字典
    """
    P, R, F1 = bert_score(hypotheses, references, lang=lang, verbose=False)
    return {'bert_score': F1.mean().item()}


def compute_all_metrics(
    hypotheses: List[str],
    references: List[str],
    tokenized_hyp: List[List[str]],
    tokenized_ref: List[List[str]]
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        hypotheses: 预测文本列表
        references: 参考文本列表
        tokenized_hyp: 分词后的预测文本
        tokenized_ref: 分词后的参考文本

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    # BLEU分数
    bleu = compute_bleu(tokenized_hyp, tokenized_ref)
    metrics.update(bleu)

    # ROUGE-L分数
    rouge = compute_rouge(hypotheses, references)
    metrics.update(rouge)

    # BERTScore（计算代价较高，可选）
    try:
        bert = compute_bert_score(hypotheses, references)
        metrics.update(bert)
    except Exception as e:
        print(f"BERTScore计算失败: {e}")
        metrics['bert_score'] = 0.0

    return metrics


def compute_diversity(texts: List[str], n_grams: List[int] = [1, 2]) -> Dict[str, float]:
    """
    计算文本多样性（distinct n-gram比例）

    Args:
        texts: 文本列表
        n_grams: 要计算的n-gram列表

    Returns:
        多样性分数字典
    """
    diversity = {}

    for n in n_grams:
        all_ngrams = []
        total_ngrams = 0

        for text in texts:
            tokens = text.split()
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
            total_ngrams += len(ngrams)

        if total_ngrams > 0:
            diversity[f'distinct_{n}'] = len(set(all_ngrams)) / total_ngrams
        else:
            diversity[f'distinct_{n}'] = 0.0

    return diversity


def compute_gradient_norm(model: nn.Module) -> Dict[str, float]:
    """
    计算模型各部分的梯度范数

    Args:
        model: PyTorch模型

    Returns:
        梯度范数字典
    """
    grad_norms = defaultdict(list)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()

            # 根据参数名称分类
            if 'encoder' in name:
                grad_norms['encoder'].append(grad_norm)
            elif 'decoder' in name:
                grad_norms['decoder'].append(grad_norm)

            if 'attention' in name or 'attn' in name:
                grad_norms['attention'].append(grad_norm)
            elif 'ffn' in name or 'linear' in name or 'fc' in name:
                grad_norms['ffn'].append(grad_norm)

    # 计算平均梯度范数
    result = {}
    for key, values in grad_norms.items():
        if values:
            result[f'grad_norm_{key}'] = np.mean(values)

    # 总体梯度范数
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    result['grad_norm_total'] = np.sqrt(total_norm)

    return result


def check_gradient_health(grad_norms: Dict[str, float]) -> Dict[str, bool]:
    """
    检查梯度是否健康（无消失/爆炸）

    Args:
        grad_norms: 梯度范数字典

    Returns:
        梯度健康状态字典
    """
    health = {}

    total_norm = grad_norms.get('grad_norm_total', 0)

    # 检查梯度消失（<1e-6）
    health['gradient_vanishing'] = total_norm < 1e-6

    # 检查梯度爆炸（>1e3）
    health['gradient_exploding'] = total_norm > 1e3

    # 总体健康状态
    health['is_healthy'] = not health['gradient_vanishing'] and not health['gradient_exploding']

    return health


class MetricsTracker:
    """指标跟踪器"""

    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}

    def update(self, metrics: Dict[str, float], epoch: int):
        """更新指标"""
        for key, value in metrics.items():
            self.metrics_history[key].append((epoch, value))

            # 更新最佳指标
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                # BLEU和BERTScore越高越好，loss越低越好
                if 'loss' in key:
                    self.best_metrics[key] = min(self.best_metrics[key], value)
                else:
                    self.best_metrics[key] = max(self.best_metrics[key], value)

    def get_history(self, key: str) -> List[Tuple[int, float]]:
        """获取指标历史"""
        return self.metrics_history.get(key, [])

    def get_best(self, key: str) -> Optional[float]:
        """获取最佳指标"""
        return self.best_metrics.get(key)

    def is_best(self, key: str, value: float, mode: str = 'max') -> bool:
        """判断是否为最佳"""
        if key not in self.best_metrics:
            return True

        if mode == 'max':
            return value > self.best_metrics[key]
        else:
            return value < self.best_metrics[key]
