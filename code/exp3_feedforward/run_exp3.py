# -*- coding: utf-8 -*-
"""
实验三：FeedForward网络结构对比实验
探究FFN的激活函数、隐藏层维度对模型特征提取能力的影响
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.base_config import DEVICE, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
from data.dataset import prepare_data
from model.base_transformer import create_transformer_model
from train.trainer import Trainer, create_optimizer_and_scheduler
from train.utils import compute_gradient_norm, check_gradient_health
from visualization.plot_utils import (
    plot_gradient_norm_curve,
    plot_training_curves,
    plot_feature_tsne,
    plot_metrics_comparison
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_experiment(config: dict, save_dir: str, data_loaders: tuple, preprocessor):
    """运行单个实验配置"""
    train_loader, val_loader, test_loader = data_loaders

    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"运行实验: {config['experiment']['name']}")
    print(f"{'='*50}")

    model_config = {**MODEL_CONFIG, **config['model']}
    train_config = {**TRAIN_CONFIG, **config['training']}

    # 创建模型
    model = create_transformer_model(
        src_vocab_size=len(preprocessor.src_vocab),
        tgt_vocab_size=len(preprocessor.tgt_vocab),
        config=model_config,
        pad_idx=preprocessor.src_vocab.pad_idx
    )

    optimizer, scheduler = create_optimizer_and_scheduler(model, train_config)

    # 自定义训练器以记录梯度范数
    trainer = GradientTrackingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_config,
        save_dir=save_dir,
        device=DEVICE,
        preprocessor=preprocessor
    )

    history = trainer.train(train_config['epochs'])

    # 绘制训练曲线
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        os.path.join(img_dir, 'training_curves.png'),
        val_metric=history['val_bleu_4'],
        metric_name='BLEU-4'
    )

    # 绘制梯度范数曲线
    if history.get('gradient_history'):
        plot_gradient_norm_curve(
            history['gradient_history'],
            os.path.join(img_dir, 'gradient_norms.png')
        )

    # 特征提取能力分析
    feature_analysis = analyze_feature_extraction(
        model, test_loader, preprocessor, DEVICE, img_dir
    )

    # 统计梯度异常
    gradient_issues = count_gradient_issues(history.get('gradient_history', []))

    results = {
        'config': config,
        'test_metrics': history['test_metrics'],
        'gradient_vanishing_epochs': gradient_issues['vanishing'],
        'gradient_exploding_epochs': gradient_issues['exploding'],
        'feature_extraction_score': feature_analysis.get('avg_similarity', 0),
        **history['test_metrics']
    }

    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    return results


class GradientTrackingTrainer(Trainer):
    """带梯度跟踪的训练器"""

    def train(self, num_epochs: int):
        """训练并记录梯度历史"""
        self.history['gradient_history'] = []
        result = super().train(num_epochs)

        # 将梯度范数历史添加到结果中
        result['gradient_history'] = self.history['gradient_history']
        return result

    def train_epoch(self, epoch: int):
        """重写以记录详细梯度信息"""
        result = super().train_epoch(epoch)

        # 记录梯度信息
        grad_norms = {k: v for k, v in result.items() if 'grad_norm' in k}
        self.history['gradient_history'].append(grad_norms)

        return result


def analyze_feature_extraction(model, test_loader, preprocessor, device, save_dir):
    """分析特征提取能力"""
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(device)

            # 获取编码器输出
            src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
            encoder_output, _ = model.encoder(src, src_mask)

            # 取平均池化作为句子表示
            mask = (src != model.pad_idx).unsqueeze(-1).float()
            pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1)

            all_features.append(pooled.cpu().numpy())

            # 使用源序列长度作为简单标签（用于聚类可视化）
            lengths = (src != model.pad_idx).sum(dim=1).cpu().numpy()
            # 将长度分为几个区间
            labels = np.digitize(lengths, bins=[5, 10, 15, 20])
            all_labels.append(labels)

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # t-SNE可视化
    if len(features) > 50:
        # 随机采样以加速
        indices = np.random.choice(len(features), min(500, len(features)), replace=False)
        sample_features = features[indices]
        sample_labels = labels[indices]

        plot_feature_tsne(
            sample_features,
            sample_labels,
            os.path.join(save_dir, 'feature_tsne.png'),
            title='Encoder Feature Visualization (t-SNE)'
        )

    # 计算特征相似度（作为特征提取能力的度量）
    # 同类样本应该有更高的相似度
    avg_similarity = compute_intra_class_similarity(features, labels)

    return {'avg_similarity': avg_similarity}


def compute_intra_class_similarity(features, labels):
    """计算类内平均相似度"""
    unique_labels = np.unique(labels)
    similarities = []

    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) > 1:
            sim_matrix = cosine_similarity(class_features)
            # 取非对角线元素的平均
            mask = ~np.eye(len(class_features), dtype=bool)
            similarities.append(sim_matrix[mask].mean())

    return np.mean(similarities) if similarities else 0


def count_gradient_issues(gradient_history):
    """统计梯度消失/爆炸的epoch数"""
    vanishing = 0
    exploding = 0

    for grad_info in gradient_history:
        total_norm = grad_info.get('grad_norm_total', 0)
        if total_norm < 1e-6:
            vanishing += 1
        if total_norm > 1e3:
            exploding += 1

    return {'vanishing': vanishing, 'exploding': exploding}


def run_experiment():
    """运行完整的FFN对比实验"""
    parser = argparse.ArgumentParser(description='实验三：FeedForward网络对比实验')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./exp3_feedforward/results')
    args = parser.parse_args()

    print("准备数据...")
    train_loader, val_loader, test_loader, preprocessor = prepare_data(
        batch_size=DATA_CONFIG['batch_size'],
        data_dir='./data_cache'
    )
    data_loaders = (train_loader, val_loader, test_loader)

    config_dir = Path(__file__).parent / 'configs'
    output_dir = Path(args.output_dir)

    if args.config:
        config_files = [config_dir / args.config]
    else:
        config_files = list(config_dir.glob('*.yaml'))

    all_results = {}

    for config_file in config_files:
        config = load_config(str(config_file))
        exp_name = config['experiment']['name']
        save_dir = output_dir / exp_name

        results = run_single_experiment(
            config, str(save_dir), data_loaders, preprocessor
        )
        all_results[exp_name] = results

    if len(all_results) > 1:
        generate_comparison_plots(all_results, str(output_dir / 'comparison'))

    with open(output_dir / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n实验完成！结果保存在: {output_dir}")


def generate_comparison_plots(all_results: dict, save_dir: str):
    """生成对比可视化"""
    os.makedirs(save_dir, exist_ok=True)

    configs = list(all_results.keys())
    metrics = {
        'BLEU-4': [all_results[c].get('bleu_4', 0) for c in configs],
        'Feature Score': [all_results[c].get('feature_extraction_score', 0) for c in configs],
        'Gradient Issues': [
            all_results[c].get('gradient_vanishing_epochs', 0) +
            all_results[c].get('gradient_exploding_epochs', 0)
            for c in configs
        ]
    }

    plot_metrics_comparison(
        configs, metrics,
        os.path.join(save_dir, 'metrics_comparison.png'),
        title='FeedForward Network Comparison'
    )


if __name__ == '__main__':
    run_experiment()
