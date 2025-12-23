# -*- coding: utf-8 -*-
"""
实验一：位置编码对比实验
探究不同位置编码方式对Transformer翻译性能的影响
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from config.base_config import DEVICE, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
from data.dataset import prepare_data
from model.base_transformer import create_transformer_model
from model.pe_module import compute_pe_similarity
from train.trainer import Trainer, create_optimizer_and_scheduler
from train.utils import compute_all_metrics
from visualization.plot_utils import (
    plot_pe_similarity_heatmap,
    plot_training_curves,
    plot_metrics_comparison,
    plot_error_type_distribution
)
from infer.decoder import greedy_decode


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_experiment(config: dict, save_dir: str, data_loaders: tuple, preprocessor):
    """运行单个实验配置"""
    train_loader, val_loader, test_loader = data_loaders

    # 创建保存目录
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

    # 训练
    history = trainer.train(train_config['epochs'])

    # 绘制训练曲线
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        os.path.join(img_dir, 'training_curves.png'),
        val_metric=history['val_bleu_4'],
        metric_name='BLEU-4'
    )

    # 绘制位置编码相似度热力图
    pe_type = config['model']['pe_type']
    pe_module = model.encoder.pos_encoder
    similarity = compute_pe_similarity(pe_module, seq_len=20)
    plot_pe_similarity_heatmap(
        similarity.cpu().numpy(),
        os.path.join(img_dir, f'pe_similarity_{pe_type}.png'),
        title=f'Position Encoding Similarity ({pe_type})'
    )

    # 评估长句翻译性能
    long_sentence_results = evaluate_long_sentences(
        model, test_loader, preprocessor, DEVICE
    )

    # 保存结果
    results = {
        'config': config,
        'test_metrics': history['test_metrics'],
        'long_sentence_accuracy': long_sentence_results['accuracy'],
        'convergence_epoch': find_convergence_epoch(history['val_bleu_4']),
        **history['test_metrics']
    }

    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def evaluate_long_sentences(model, test_loader, preprocessor, device, length_threshold=15):
    """评估长句（长度>15token）翻译性能"""
    model.eval()

    long_predictions = []
    long_references = []
    error_types = {'语序错误': 0, '词汇缺失': 0, '语法错误': 0}

    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # 筛选长句
            src_lengths = (src != preprocessor.src_vocab.pad_idx).sum(dim=1)

            for i in range(src.size(0)):
                if src_lengths[i] > length_threshold:
                    # 生成预测
                    pred = greedy_decode(
                        model, src[i:i+1],
                        preprocessor.tgt_vocab.sos_idx,
                        preprocessor.tgt_vocab.eos_idx,
                        max_len=DATA_CONFIG['max_seq_len'],
                        device=device
                    )

                    # 解码
                    pred_tokens = []
                    for idx in pred[0].tolist():
                        if idx == preprocessor.tgt_vocab.eos_idx:
                            break
                        if idx not in [preprocessor.tgt_vocab.pad_idx, preprocessor.tgt_vocab.sos_idx]:
                            pred_tokens.append(preprocessor.tgt_vocab.idx2word.get(idx, '<unk>'))

                    ref_tokens = []
                    for idx in tgt[i].tolist():
                        if idx == preprocessor.tgt_vocab.eos_idx:
                            break
                        if idx not in [preprocessor.tgt_vocab.pad_idx, preprocessor.tgt_vocab.sos_idx]:
                            ref_tokens.append(preprocessor.tgt_vocab.idx2word.get(idx, '<unk>'))

                    long_predictions.append(pred_tokens)
                    long_references.append(ref_tokens)

                    # 简单错误分类
                    classify_error(pred_tokens, ref_tokens, error_types)

    # 计算BLEU
    if long_predictions:
        from train.utils import compute_bleu
        bleu = compute_bleu(long_predictions, long_references)
        accuracy = bleu.get('bleu_4', 0)
    else:
        accuracy = 0

    return {
        'accuracy': accuracy,
        'error_types': error_types,
        'num_long_sentences': len(long_predictions)
    }


def classify_error(pred_tokens, ref_tokens, error_types):
    """简单错误分类"""
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)

    missing = ref_set - pred_set
    extra = pred_set - ref_set

    if len(missing) > len(ref_set) * 0.3:
        error_types['词汇缺失'] += 1
    elif pred_tokens != ref_tokens and len(missing) <= 2:
        error_types['语序错误'] += 1
    elif len(extra) > 0:
        error_types['语法错误'] += 1


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


def run_experiment():
    """运行完整的位置编码对比实验"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='实验一：位置编码对比实验')
    parser.add_argument('--config', type=str, default=None, help='指定单个配置文件')
    parser.add_argument('--output_dir', type=str, default='./exp1_position_encoding/results',
                        help='输出目录')
    args = parser.parse_args()

    # 准备数据
    print("准备数据...")
    train_loader, val_loader, test_loader, preprocessor = prepare_data(
        batch_size=DATA_CONFIG['batch_size'],
        data_dir='./data_cache'
    )
    data_loaders = (train_loader, val_loader, test_loader)

    # 配置文件目录
    config_dir = Path(__file__).parent / 'configs'
    output_dir = Path(args.output_dir)

    if args.config:
        # 运行单个配置
        config_files = [config_dir / args.config]
    else:
        # 运行所有配置
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
        'BLEU-1': [all_results[c].get('bleu_1', 0) for c in configs],
        'BLEU-2': [all_results[c].get('bleu_2', 0) for c in configs],
        'BLEU-4': [all_results[c].get('bleu_4', 0) for c in configs],
        'ROUGE-L': [all_results[c].get('rouge_l', 0) for c in configs]
    }

    plot_metrics_comparison(
        configs, metrics,
        os.path.join(save_dir, 'metrics_comparison.png'),
        title='Position Encoding Methods Comparison'
    )

    # 收敛速度对比
    convergence_data = {
        '收敛轮次': [all_results[c].get('convergence_epoch', 0) for c in configs]
    }
    plot_metrics_comparison(
        configs, convergence_data,
        os.path.join(save_dir, 'convergence_comparison.png'),
        title='Convergence Speed Comparison'
    )


if __name__ == '__main__':
    run_experiment()
