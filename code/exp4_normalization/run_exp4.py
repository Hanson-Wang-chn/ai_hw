# -*- coding: utf-8 -*-
"""
实验四：归一化策略对比实验
验证Pre-LN与Post-LN对训练稳定性和收敛速度的影响
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

from config.base_config import DEVICE, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
from data.dataset import prepare_data
from model.base_transformer import create_transformer_model, TransformerSeq2Seq
from train.trainer import Trainer, create_optimizer_and_scheduler
from visualization.plot_utils import (
    plot_training_curves,
    plot_training_curves_with_ci,
    plot_metrics_comparison
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_mixed_norm_model(src_vocab_size, tgt_vocab_size, config, pad_idx):
    """创建混合归一化模型（Encoder用Pre-LN，Decoder用Post-LN）"""
    # 先创建标准模型
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
        norm_type='pre',  # 先用pre创建
        attn_type=config.get('attn_type', 'multi_head'),
        max_len=config.get('max_len', 5000),
        pad_idx=pad_idx
    )

    # 修改decoder层的归一化策略为post
    from model.base_transformer import TransformerDecoderLayer
    for i, layer in enumerate(model.decoder.layers):
        layer.norm_type = 'post'

    print(f"混合归一化模型参数量: {model.count_parameters():,}")
    return model


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

    # 根据norm_type创建模型
    if model_config.get('norm_type') == 'mixed':
        model = create_mixed_norm_model(
            src_vocab_size=len(preprocessor.src_vocab),
            tgt_vocab_size=len(preprocessor.tgt_vocab),
            config=model_config,
            pad_idx=preprocessor.src_vocab.pad_idx
        )
    else:
        model = create_transformer_model(
            src_vocab_size=len(preprocessor.src_vocab),
            tgt_vocab_size=len(preprocessor.tgt_vocab),
            config=model_config,
            pad_idx=preprocessor.src_vocab.pad_idx
        )

    optimizer, scheduler = create_optimizer_and_scheduler(model, train_config)

    trainer = Trainer(
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

    # 计算训练稳定性（损失方差）
    loss_variance = np.var(history['train_loss'])

    # 找到收敛epoch
    convergence_epoch = find_convergence_epoch(history['val_bleu_4'])

    results = {
        'config': config,
        'test_metrics': history['test_metrics'],
        'loss_variance': float(loss_variance),
        'convergence_epoch': convergence_epoch,
        'train_loss_history': history['train_loss'],
        'val_loss_history': history['val_loss'],
        **history['test_metrics']
    }

    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def find_convergence_epoch(bleu_history, threshold=0.8):
    """找到达到最佳BLEU 80%的epoch"""
    if not bleu_history:
        return -1
    max_bleu = max(bleu_history)
    target = max_bleu * threshold
    for epoch, bleu in enumerate(bleu_history, 1):
        if bleu >= target:
            return epoch
    return len(bleu_history)


def test_learning_rate_sensitivity(model, train_loader, val_loader, preprocessor, device, save_dir):
    """测试不同学习率下的性能"""
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    results = {}

    for lr in learning_rates:
        # 重置模型（简化处理：只测试少量epoch）
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # 训练几个epoch
        model.train()
        losses = []
        for epoch in range(3):
            epoch_loss = 0
            for batch in train_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                optimizer.zero_grad()
                outputs = model(src, tgt)
                loss = torch.nn.functional.cross_entropy(
                    outputs['output'][:, 1:].reshape(-1, outputs['output'].size(-1)),
                    tgt[:, 1:].reshape(-1),
                    ignore_index=preprocessor.src_vocab.pad_idx
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(train_loader))

        results[lr] = losses[-1]

    return results


def run_experiment():
    """运行完整的归一化策略对比实验"""
    parser = argparse.ArgumentParser(description='实验四：归一化策略对比实验')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./exp4_normalization/results')
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
        'Loss Variance': [all_results[c].get('loss_variance', 0) for c in configs],
        'Convergence Epoch': [all_results[c].get('convergence_epoch', 0) for c in configs]
    }

    plot_metrics_comparison(
        configs, metrics,
        os.path.join(save_dir, 'metrics_comparison.png'),
        title='Normalization Strategy Comparison'
    )

    # 绘制带置信区间的训练曲线对比
    train_losses = []
    val_losses = []
    labels = []

    for config_name, results in all_results.items():
        if 'train_loss_history' in results:
            train_losses.append([results['train_loss_history']])
            val_losses.append([results['val_loss_history']])
            labels.append(config_name)

    if train_losses:
        plot_training_curves_with_ci(
            train_losses, val_losses,
            os.path.join(save_dir, 'training_curves_comparison.png'),
            labels=labels
        )


if __name__ == '__main__':
    run_experiment()
