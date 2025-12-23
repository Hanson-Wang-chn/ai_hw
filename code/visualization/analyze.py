# -*- coding: utf-8 -*-
"""
结果分析脚本
生成分析报告和可视化
"""

import os
import json
from typing import Dict, List, Optional
import numpy as np
import torch
from collections import defaultdict

from visualization.plot_utils import (
    plot_attention_heatmap,
    plot_multi_head_attention,
    plot_pe_similarity_heatmap,
    plot_gradient_norm_curve,
    plot_training_curves,
    plot_feature_tsne,
    plot_metrics_comparison,
    plot_beam_width_performance,
    plot_error_type_distribution,
    plot_diversity_vs_quality
)


class ResultAnalyzer:
    """结果分析器"""

    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 结果目录
        """
        self.results_dir = results_dir
        self.analysis_dir = os.path.join(results_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)

    def load_experiment_results(self, exp_name: str) -> Dict:
        """加载实验结果"""
        result_path = os.path.join(self.results_dir, exp_name, 'results.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                return json.load(f)
        return {}

    def analyze_position_encoding(self, results: Dict):
        """分析位置编码实验结果"""
        save_dir = os.path.join(self.analysis_dir, 'position_encoding')
        os.makedirs(save_dir, exist_ok=True)

        configs = list(results.keys())
        metrics = defaultdict(list)

        for config in configs:
            config_results = results[config]
            metrics['BLEU-1'].append(config_results.get('bleu_1', 0))
            metrics['BLEU-2'].append(config_results.get('bleu_2', 0))
            metrics['BLEU-4'].append(config_results.get('bleu_4', 0))
            metrics['ROUGE-L'].append(config_results.get('rouge_l', 0))

        # 绘制指标对比图
        plot_metrics_comparison(
            configs, dict(metrics),
            os.path.join(save_dir, 'metrics_comparison.png'),
            title='Position Encoding Comparison'
        )

        # 分析长句翻译性能
        long_sentence_accuracy = {}
        for config in configs:
            config_results = results[config]
            long_sentence_accuracy[config] = config_results.get('long_sentence_accuracy', 0)

        # 生成分析报告
        report = self._generate_pe_report(results, metrics)
        with open(os.path.join(save_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)

    def analyze_attention(self, results: Dict):
        """分析注意力机制实验结果"""
        save_dir = os.path.join(self.analysis_dir, 'attention')
        os.makedirs(save_dir, exist_ok=True)

        configs = list(results.keys())
        metrics = defaultdict(list)

        for config in configs:
            config_results = results[config]
            metrics['BLEU-4'].append(config_results.get('bleu_4', 0))
            metrics['Attention Entropy'].append(config_results.get('attention_entropy', 0))
            metrics['Training Time'].append(config_results.get('training_time', 0))

        # 绘制指标对比图
        plot_metrics_comparison(
            configs, dict(metrics),
            os.path.join(save_dir, 'metrics_comparison.png'),
            title='Attention Mechanism Comparison'
        )

    def analyze_feedforward(self, results: Dict):
        """分析FFN实验结果"""
        save_dir = os.path.join(self.analysis_dir, 'feedforward')
        os.makedirs(save_dir, exist_ok=True)

        configs = list(results.keys())

        # 收集梯度范数历史
        for config in configs:
            config_results = results[config]
            if 'gradient_history' in config_results:
                plot_gradient_norm_curve(
                    config_results['gradient_history'],
                    os.path.join(save_dir, f'gradient_norm_{config}.png')
                )

    def analyze_decoding(self, results: Dict):
        """分析解码策略实验结果"""
        save_dir = os.path.join(self.analysis_dir, 'decoding')
        os.makedirs(save_dir, exist_ok=True)

        # 束搜索性能分析
        if 'beam_search' in results:
            beam_results = results['beam_search']
            beam_widths = list(beam_results.keys())
            bleu_scores = [beam_results[bw]['bleu_4'] for bw in beam_widths]
            decode_times = [beam_results[bw]['decode_time'] for bw in beam_widths]

            plot_beam_width_performance(
                [int(bw) for bw in beam_widths],
                bleu_scores,
                decode_times,
                os.path.join(save_dir, 'beam_width_performance.png')
            )

        # 多样性vs质量分析
        if 'sampling' in results:
            sampling_results = results['sampling']
            diversity_scores = []
            quality_scores = []
            labels = []

            for temp, temp_results in sampling_results.items():
                diversity_scores.append(temp_results.get('distinct_2', 0))
                quality_scores.append(temp_results.get('bleu_4', 0))
                labels.append(f'T={temp}')

            plot_diversity_vs_quality(
                diversity_scores, quality_scores, labels,
                os.path.join(save_dir, 'diversity_vs_quality.png')
            )

    def analyze_error_types(
        self,
        predictions: List[str],
        references: List[str],
        save_dir: str
    ):
        """
        分析翻译错误类型

        Args:
            predictions: 预测翻译列表
            references: 参考翻译列表
            save_dir: 保存目录
        """
        error_counts = {
            '语序错误': 0,
            '词汇缺失': 0,
            '语法错误': 0,
            '其他': 0
        }

        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            # 简化的错误分类
            missing = ref_tokens - pred_tokens
            extra = pred_tokens - ref_tokens

            if len(missing) > len(ref_tokens) * 0.3:
                error_counts['词汇缺失'] += 1
            elif len(extra) > len(pred_tokens) * 0.3:
                error_counts['语法错误'] += 1
            elif pred != ref and len(missing) < 3:
                error_counts['语序错误'] += 1
            elif pred != ref:
                error_counts['其他'] += 1

        plot_error_type_distribution(
            error_counts,
            os.path.join(save_dir, 'error_types.png'),
            title='Translation Error Type Distribution'
        )

        return error_counts

    def _generate_pe_report(self, results: Dict, metrics: Dict) -> str:
        """生成位置编码实验分析报告"""
        report = []
        report.append("=" * 50)
        report.append("位置编码实验分析报告")
        report.append("=" * 50)
        report.append("")

        # 最佳配置
        best_config = max(results.keys(), key=lambda k: results[k].get('bleu_4', 0))
        best_bleu = results[best_config].get('bleu_4', 0)

        report.append(f"最佳配置: {best_config}")
        report.append(f"最佳BLEU-4: {best_bleu:.4f}")
        report.append("")

        # 各配置对比
        report.append("各配置详细结果:")
        report.append("-" * 40)
        for config, config_results in results.items():
            report.append(f"\n{config}:")
            report.append(f"  BLEU-1: {config_results.get('bleu_1', 0):.4f}")
            report.append(f"  BLEU-2: {config_results.get('bleu_2', 0):.4f}")
            report.append(f"  BLEU-4: {config_results.get('bleu_4', 0):.4f}")
            report.append(f"  ROUGE-L: {config_results.get('rouge_l', 0):.4f}")
            report.append(f"  收敛轮次: {config_results.get('convergence_epoch', 'N/A')}")

        # 结论
        report.append("")
        report.append("分析结论:")
        report.append("-" * 40)
        report.append("1. 位置编码方式对翻译性能有显著影响")
        report.append(f"2. {best_config}在本实验中表现最佳")
        report.append("3. 建议在类似短文本翻译任务中优先考虑该配置")

        return "\n".join(report)

    def generate_comprehensive_report(self, all_results: Dict):
        """生成综合分析报告"""
        report_path = os.path.join(self.analysis_dir, 'comprehensive_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Seq2Seq模型实验综合分析报告\n\n")

            for exp_name, results in all_results.items():
                f.write(f"## {exp_name}\n\n")

                if isinstance(results, dict):
                    for config, config_results in results.items():
                        f.write(f"### {config}\n\n")
                        if isinstance(config_results, dict):
                            for metric, value in config_results.items():
                                if isinstance(value, (int, float)):
                                    f.write(f"- {metric}: {value:.4f}\n")
                        f.write("\n")

                f.write("---\n\n")

        print(f"综合报告已生成: {report_path}")


def extract_attention_weights(model, src, tgt, device):
    """
    提取模型的注意力权重

    Args:
        model: Transformer模型
        src: 源序列
        tgt: 目标序列
        device: 计算设备

    Returns:
        注意力权重字典
    """
    model.eval()
    src = src.to(device)
    tgt = tgt.to(device)

    with torch.no_grad():
        outputs = model(src, tgt, return_attention=True)

    return {
        'encoder_attention': outputs.get('encoder_attention'),
        'decoder_self_attention': outputs.get('decoder_self_attention'),
        'decoder_cross_attention': outputs.get('decoder_cross_attention')
    }


def extract_encoder_features(model, src, device):
    """
    提取编码器特征

    Args:
        model: Transformer模型
        src: 源序列
        device: 计算设备

    Returns:
        编码器输出特征
    """
    model.eval()
    src = src.to(device)

    with torch.no_grad():
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        encoder_output, _ = model.encoder(src, src_mask)

    return encoder_output.cpu().numpy()
