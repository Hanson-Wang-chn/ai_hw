# -*- coding: utf-8 -*-
"""
实验二：注意力机制变体对比实验
分析不同注意力机制对模型捕捉源-目标语言语义关联的影响
"""

import os
import sys
import json
import yaml
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from config.base_config import DEVICE, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
from data.dataset import prepare_data
from model.base_transformer import create_transformer_model
from model.attention_module import compute_attention_entropy
from train.trainer import Trainer, create_optimizer_and_scheduler
from visualization.plot_utils import (
    plot_attention_heatmap,
    plot_multi_head_attention,
    plot_training_curves,
    plot_metrics_comparison,
    plot_attention_entropy_curve
)
from infer.decoder import greedy_decode


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
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

    # 合并配置
    model_config = {**MODEL_CONFIG, **config['model']}
    train_config = {**TRAIN_CONFIG, **config['training']}

    # 创建模型
    model = create_transformer_model(
        src_vocab_size=len(preprocessor.src_vocab),
        tgt_vocab_size=len(preprocessor.tgt_vocab),
        config=model_config,
        pad_idx=preprocessor.src_vocab.pad_idx
    )

    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_config)

    # 创建训练器
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

    # 记录训练时间
    start_time = time.time()
    history = trainer.train(train_config['epochs'])
    training_time = time.time() - start_time

    # 绘制训练曲线
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        os.path.join(img_dir, 'training_curves.png'),
        val_metric=history['val_bleu_4'],
        metric_name='BLEU-4'
    )

    # 提取并可视化注意力权重
    attention_analysis = analyze_attention(
        model, test_loader, preprocessor, DEVICE, img_dir
    )

    # 保存结果
    results = {
        'config': config,
        'test_metrics': history['test_metrics'],
        'training_time': training_time,
        'attention_entropy': attention_analysis['avg_entropy'],
        **history['test_metrics']
    }

    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def analyze_attention(model, test_loader, preprocessor, device, save_dir):
    """分析注意力权重"""
    model.eval()
    all_entropies = []

    # 获取一些样本的注意力权重
    sample_count = 0
    max_samples = 5

    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # 获取注意力权重
            outputs = model(src, tgt, return_attention=True)

            # 计算注意力熵
            cross_attn = outputs.get('decoder_cross_attention')
            if cross_attn and len(cross_attn) > 0:
                # 取最后一层的注意力权重
                last_layer_attn = cross_attn[-1]
                if last_layer_attn is not None:
                    entropy = compute_attention_entropy(last_layer_attn)
                    all_entropies.append(entropy.item())

            # 可视化前几个样本的注意力
            if sample_count < max_samples and cross_attn and cross_attn[-1] is not None:
                attn_weights = cross_attn[-1][0].cpu().numpy()  # 第一个样本

                # 获取源和目标tokens
                src_tokens = decode_tokens(src[0], preprocessor.src_vocab)
                tgt_tokens = decode_tokens(tgt[0], preprocessor.tgt_vocab)

                # 如果是多头注意力，绘制多头可视化
                if len(attn_weights.shape) == 3:
                    plot_multi_head_attention(
                        attn_weights,
                        src_tokens[:attn_weights.shape[-1]],
                        tgt_tokens[:attn_weights.shape[-2]],
                        os.path.join(save_dir, f'attention_sample_{sample_count}.png'),
                        num_heads=attn_weights.shape[0]
                    )
                else:
                    plot_attention_heatmap(
                        attn_weights,
                        src_tokens[:attn_weights.shape[-1]],
                        tgt_tokens[:attn_weights.shape[-2]],
                        os.path.join(save_dir, f'attention_sample_{sample_count}.png')
                    )

                sample_count += 1

            if sample_count >= max_samples:
                break

    return {
        'avg_entropy': np.mean(all_entropies) if all_entropies else 0,
        'entropy_std': np.std(all_entropies) if all_entropies else 0
    }


def decode_tokens(indices, vocab):
    """将索引转换为token"""
    tokens = []
    for idx in indices.tolist():
        if idx == vocab.eos_idx:
            break
        if idx not in [vocab.pad_idx, vocab.sos_idx]:
            tokens.append(vocab.idx2word.get(idx, '<unk>'))
    return tokens


def run_experiment():
    """运行完整的注意力机制对比实验"""
    parser = argparse.ArgumentParser(description='实验二：注意力机制对比实验')
    parser.add_argument('--config', type=str, default=None, help='指定单个配置文件')
    parser.add_argument('--output_dir', type=str, default='./exp2_attention/results',
                        help='输出目录')
    args = parser.parse_args()

    # 准备数据
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
    entropy_history = {}

    for config_file in config_files:
        config = load_config(str(config_file))
        exp_name = config['experiment']['name']
        save_dir = output_dir / exp_name

        results = run_single_experiment(
            config, str(save_dir), data_loaders, preprocessor
        )
        all_results[exp_name] = results

    # 生成对比可视化
    if len(all_results) > 1:
        generate_comparison_plots(all_results, str(output_dir / 'comparison'))

    # 保存汇总结果
    with open(output_dir / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n实验完成！结果保存在: {output_dir}")


def generate_comparison_plots(all_results: dict, save_dir: str):
    """生成对比可视化"""
    os.makedirs(save_dir, exist_ok=True)

    configs = list(all_results.keys())
    metrics = {
        'BLEU-4': [all_results[c].get('bleu_4', 0) for c in configs],
        'Attention Entropy': [all_results[c].get('attention_entropy', 0) for c in configs],
        'Training Time (s)': [all_results[c].get('training_time', 0) for c in configs]
    }

    plot_metrics_comparison(
        configs, metrics,
        os.path.join(save_dir, 'metrics_comparison.png'),
        title='Attention Mechanism Comparison'
    )


if __name__ == '__main__':
    run_experiment()
