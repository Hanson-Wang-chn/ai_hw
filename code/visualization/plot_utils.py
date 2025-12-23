# -*- coding: utf-8 -*-
"""
可视化工具函数
包含注意力热力图、梯度曲线、特征聚类等可视化功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Tuple
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style('whitegrid')


def save_figure(fig, save_path: str, dpi: int = 150):
    """保存图片"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"图片已保存至: {save_path}")


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    src_tokens: List[str],
    tgt_tokens: List[str],
    save_path: str,
    title: str = 'Attention Weights'
):
    """
    绘制注意力热力图

    Args:
        attention_weights: 注意力权重矩阵 (tgt_len, src_len)
        src_tokens: 源序列token列表
        tgt_tokens: 目标序列token列表
        save_path: 保存路径
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        attention_weights,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        ax=ax
    )

    ax.set_xlabel('Source Tokens')
    ax.set_ylabel('Target Tokens')
    ax.set_title(title)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    save_figure(fig, save_path)


def plot_multi_head_attention(
    attention_weights: np.ndarray,
    src_tokens: List[str],
    tgt_tokens: List[str],
    save_path: str,
    num_heads: int = 8
):
    """
    绘制多头注意力热力图

    Args:
        attention_weights: 注意力权重 (num_heads, tgt_len, src_len)
        src_tokens: 源序列token列表
        tgt_tokens: 目标序列token列表
        save_path: 保存路径
        num_heads: 注意力头数
    """
    rows = 2
    cols = (num_heads + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for head in range(num_heads):
        ax = axes[head]
        sns.heatmap(
            attention_weights[head],
            xticklabels=src_tokens if head >= cols else [],
            yticklabels=tgt_tokens if head % cols == 0 else [],
            cmap='Blues',
            ax=ax,
            cbar=False
        )
        ax.set_title(f'Head {head + 1}')

    plt.suptitle('Multi-Head Attention Visualization')
    plt.tight_layout()

    save_figure(fig, save_path)


def plot_pe_similarity_heatmap(
    similarity_matrix: np.ndarray,
    save_path: str,
    title: str = 'Position Encoding Similarity'
):
    """
    绘制位置编码相似度热力图

    Args:
        similarity_matrix: 相似度矩阵 (seq_len, seq_len)
        save_path: 保存路径
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        similarity_matrix,
        cmap='RdYlBu_r',
        center=0,
        ax=ax,
        square=True
    )

    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title(title)

    save_figure(fig, save_path)


def plot_gradient_norm_curve(
    gradient_history: List[Dict[str, float]],
    save_path: str,
    keys: List[str] = None
):
    """
    绘制梯度范数演化曲线

    Args:
        gradient_history: 梯度历史列表
        save_path: 保存路径
        keys: 要绘制的键列表
    """
    if keys is None:
        keys = ['grad_norm_encoder', 'grad_norm_decoder', 'grad_norm_total']

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(gradient_history) + 1)

    for key in keys:
        values = [g.get(key, 0) for g in gradient_history]
        ax.plot(epochs, values, label=key.replace('grad_norm_', ''), linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Evolution')
    ax.legend()
    ax.set_yscale('log')

    # 添加梯度消失/爆炸参考线
    ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Vanishing threshold')
    ax.axhline(y=1e3, color='r', linestyle='--', alpha=0.5, label='Exploding threshold')

    save_figure(fig, save_path)


def plot_training_curves(
    train_loss: List[float],
    val_loss: List[float],
    save_path: str,
    val_metric: List[float] = None,
    metric_name: str = 'BLEU-4'
):
    """
    绘制训练/验证损失曲线

    Args:
        train_loss: 训练损失列表
        val_loss: 验证损失列表
        save_path: 保存路径
        val_metric: 验证指标列表
        metric_name: 指标名称
    """
    fig, axes = plt.subplots(1, 2 if val_metric else 1, figsize=(14 if val_metric else 7, 5))

    if val_metric:
        ax1, ax2 = axes
    else:
        ax1 = axes

    epochs = range(1, len(train_loss) + 1)

    # 损失曲线
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 指标曲线
    if val_metric:
        ax2.plot(epochs, val_metric, 'g-', label=metric_name, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Validation {metric_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_training_curves_with_ci(
    train_losses: List[List[float]],
    val_losses: List[List[float]],
    save_path: str,
    labels: List[str] = None
):
    """
    绘制带置信区间的训练曲线

    Args:
        train_losses: 多次运行的训练损失列表
        val_losses: 多次运行的验证损失列表
        save_path: 保存路径
        labels: 标签列表
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for i, (train_loss_runs, val_loss_runs) in enumerate(zip(train_losses, val_losses)):
        train_mean = np.mean(train_loss_runs, axis=0)
        train_std = np.std(train_loss_runs, axis=0)
        val_mean = np.mean(val_loss_runs, axis=0)
        val_std = np.std(val_loss_runs, axis=0)

        epochs = range(1, len(train_mean) + 1)
        label = labels[i] if labels else f'Config {i + 1}'
        color = colors[i % len(colors)]

        ax.plot(epochs, val_mean, '-', color=color, label=label, linewidth=2)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color=color, alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Curves with Confidence Interval')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, save_path)


def plot_feature_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = 'Feature Visualization (t-SNE)',
    perplexity: int = 30
):
    """
    绘制特征t-SNE降维图

    Args:
        features: 特征矩阵 (num_samples, feature_dim)
        labels: 标签数组
        save_path: 保存路径
        title: 图标题
        perplexity: t-SNE perplexity参数
    """
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=50
    )

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(title)

    plt.colorbar(scatter, ax=ax, label='Label')

    save_figure(fig, save_path)


def plot_attention_entropy_curve(
    entropy_history: Dict[str, List[float]],
    save_path: str
):
    """
    绘制注意力熵演化曲线

    Args:
        entropy_history: 不同配置的注意力熵历史
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for config_name, entropy_values in entropy_history.items():
        epochs = range(1, len(entropy_values) + 1)
        ax.plot(epochs, entropy_values, label=config_name, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Entropy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, save_path)


def plot_decoding_probability(
    probs: np.ndarray,
    token_labels: List[str],
    save_path: str,
    top_k: int = 10,
    title: str = 'Decoding Probability Distribution'
):
    """
    绘制解码概率分布图

    Args:
        probs: 概率分布 (vocab_size,)
        token_labels: 词汇表标签
        save_path: 保存路径
        top_k: 显示前k个
        title: 图标题
    """
    # 获取top-k
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_labels = [token_labels[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(top_k), top_probs, color='steelblue')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_labels)
    ax.set_xlabel('Probability')
    ax.set_title(title)

    # 添加概率值标签
    for bar, prob in zip(bars, top_probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{prob:.3f}', va='center')

    ax.invert_yaxis()

    save_figure(fig, save_path)


def plot_beam_width_performance(
    beam_widths: List[int],
    bleu_scores: List[float],
    decode_times: List[float],
    save_path: str
):
    """
    绘制束宽-性能曲线

    Args:
        beam_widths: 束宽列表
        bleu_scores: BLEU分数列表
        decode_times: 解码时间列表
        save_path: 保存路径
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Beam Width')
    ax1.set_ylabel('BLEU-4', color=color1)
    line1 = ax1.plot(beam_widths, bleu_scores, 'o-', color=color1, label='BLEU-4', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Decode Time (s)', color=color2)
    line2 = ax2.plot(beam_widths, decode_times, 's--', color=color2, label='Decode Time', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    ax1.set_title('Beam Width vs Performance')
    ax1.grid(True, alpha=0.3)

    save_figure(fig, save_path)


def plot_error_type_distribution(
    error_counts: Dict[str, int],
    save_path: str,
    title: str = 'Translation Error Types'
):
    """
    绘制错误类型统计柱状图

    Args:
        error_counts: 错误类型计数字典
        save_path: 保存路径
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    types = list(error_counts.keys())
    counts = list(error_counts.values())

    bars = ax.bar(types, counts, color='coral')
    ax.set_xlabel('Error Type')
    ax.set_ylabel('Count')
    ax.set_title(title)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')

    save_figure(fig, save_path)


def plot_metrics_comparison(
    configs: List[str],
    metrics: Dict[str, List[float]],
    save_path: str,
    title: str = 'Metrics Comparison'
):
    """
    绘制不同配置的指标对比图

    Args:
        configs: 配置名称列表
        metrics: 指标字典，键为指标名，值为各配置的分数
        save_path: 保存路径
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.8 / len(metrics)

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, save_path)


def plot_diversity_vs_quality(
    diversity_scores: List[float],
    quality_scores: List[float],
    labels: List[str],
    save_path: str
):
    """
    绘制多样性vs质量散点图

    Args:
        diversity_scores: 多样性分数
        quality_scores: 质量分数
        labels: 点标签
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(diversity_scores, quality_scores, s=100, alpha=0.7)

    # 添加标签
    for i, label in enumerate(labels):
        ax.annotate(label, (diversity_scores[i], quality_scores[i]),
                    xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Diversity (Distinct-2)')
    ax.set_ylabel('Quality (BLEU-4)')
    ax.set_title('Generation Diversity vs Translation Quality')
    ax.grid(True, alpha=0.3)

    save_figure(fig, save_path)


def plot_layer_gradient_heatmap(
    layer_gradients: Dict[str, List[float]],
    save_path: str,
    title: str = 'Layer Gradient Norms'
):
    """
    绘制各层梯度范数热力图

    Args:
        layer_gradients: 各层梯度范数，键为层类型，值为各层的梯度范数
        save_path: 保存路径
        title: 图标题
    """
    # 构建矩阵
    max_len = max(len(v) for v in layer_gradients.values())
    layer_names = list(layer_gradients.keys())
    matrix = np.zeros((len(layer_names), max_len))

    for i, (name, values) in enumerate(layer_gradients.items()):
        matrix[i, :len(values)] = values

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        matrix,
        xticklabels=range(1, max_len + 1),
        yticklabels=layer_names,
        cmap='YlOrRd',
        ax=ax,
        annot=True,
        fmt='.3f'
    )

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Layer Type')
    ax.set_title(title)

    save_figure(fig, save_path)
