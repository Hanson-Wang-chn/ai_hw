# -*- coding: utf-8 -*-
"""
实验五：解码策略优化实验
探究不同解码策略对生成文本质量的影响
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
from train.trainer import Trainer, create_optimizer_and_scheduler
from train.utils import compute_bleu, compute_diversity
from infer.decoder import (
    greedy_decode,
    beam_search_decode,
    temperature_sampling_decode,
    get_decoding_probability_distribution
)
from visualization.plot_utils import (
    plot_beam_width_performance,
    plot_diversity_vs_quality,
    plot_decoding_probability,
    plot_metrics_comparison
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_base_model(data_loaders, preprocessor, config, save_dir):
    """训练基础模型"""
    train_loader, val_loader, test_loader = data_loaders

    model_config = {**MODEL_CONFIG, **config['model']}
    train_config = {**TRAIN_CONFIG, **config['training']}

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

    trainer.train(train_config['epochs'])

    return model


def evaluate_decoding_strategy(
    model,
    test_loader,
    preprocessor,
    device,
    strategy: str,
    **kwargs
):
    """评估单个解码策略"""
    model.eval()

    all_predictions = []
    all_references = []
    all_pred_tokens = []
    all_ref_tokens = []
    total_time = 0
    total_tokens = 0

    sos_idx = preprocessor.tgt_vocab.sos_idx
    eos_idx = preprocessor.tgt_vocab.eos_idx
    max_len = DATA_CONFIG['max_seq_len']

    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            start_time = time.time()

            # 根据策略选择解码方法
            if strategy == 'greedy':
                predictions = greedy_decode(
                    model, src, sos_idx, eos_idx, max_len, device
                )
                scores = None
            elif strategy == 'beam':
                beam_width = kwargs.get('beam_width', 4)
                predictions, scores = beam_search_decode(
                    model, src, sos_idx, eos_idx,
                    beam_width=beam_width,
                    max_len=max_len,
                    device=device
                )
            elif strategy == 'sampling':
                temperature = kwargs.get('temperature', 1.0)
                predictions, probs = temperature_sampling_decode(
                    model, src, sos_idx, eos_idx,
                    temperature=temperature,
                    max_len=max_len,
                    device=device
                )
            else:
                raise ValueError(f"未知的解码策略: {strategy}")

            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_tokens += predictions.numel()

            # 解码预测和参考
            for pred, ref in zip(predictions, tgt):
                pred_tokens = decode_tokens(pred, preprocessor.tgt_vocab)
                ref_tokens = decode_tokens(ref, preprocessor.tgt_vocab)

                all_pred_tokens.append(pred_tokens)
                all_ref_tokens.append(ref_tokens)
                all_predictions.append(' '.join(pred_tokens))
                all_references.append(' '.join(ref_tokens))

    # 计算指标
    bleu = compute_bleu(all_pred_tokens, all_ref_tokens)
    diversity = compute_diversity(all_predictions)

    return {
        'strategy': strategy,
        'params': kwargs,
        **bleu,
        **diversity,
        'decode_time': total_time,
        'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
        'predictions': all_predictions[:10],  # 保存部分样本
        'references': all_references[:10]
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
    """运行完整的解码策略实验"""
    parser = argparse.ArgumentParser(description='实验五：解码策略优化实验')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径（如果不提供则重新训练）')
    parser.add_argument('--output_dir', type=str, default='./exp5_decoding/results')
    args = parser.parse_args()

    print("准备数据...")
    train_loader, val_loader, test_loader, preprocessor = prepare_data(
        batch_size=DATA_CONFIG['batch_size'],
        data_dir='./data_cache'
    )
    data_loaders = (train_loader, val_loader, test_loader)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    img_dir = output_dir / 'img'
    os.makedirs(img_dir, exist_ok=True)

    # 加载或训练基础模型
    if args.model_path and os.path.exists(args.model_path):
        print(f"加载预训练模型: {args.model_path}")
        config = load_config(str(Path(__file__).parent / 'configs' / 'greedy.yaml'))
        model_config = {**MODEL_CONFIG, **config['model']}

        model = create_transformer_model(
            src_vocab_size=len(preprocessor.src_vocab),
            tgt_vocab_size=len(preprocessor.tgt_vocab),
            config=model_config,
            pad_idx=preprocessor.src_vocab.pad_idx
        )
        checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
    else:
        print("训练基础模型...")
        config = load_config(str(Path(__file__).parent / 'configs' / 'greedy.yaml'))
        model = train_base_model(
            data_loaders, preprocessor, config,
            str(output_dir / 'base_model')
        )

    all_results = {}

    # 1. 贪心搜索
    print("\n评估贪心搜索...")
    greedy_results = evaluate_decoding_strategy(
        model, test_loader, preprocessor, DEVICE, 'greedy'
    )
    all_results['greedy'] = greedy_results

    # 2. 束搜索（多种束宽）
    print("\n评估束搜索...")
    beam_results = {}
    beam_widths = [2, 4, 6, 8]
    for bw in beam_widths:
        print(f"  束宽={bw}")
        results = evaluate_decoding_strategy(
            model, test_loader, preprocessor, DEVICE,
            'beam', beam_width=bw
        )
        beam_results[str(bw)] = results

    all_results['beam_search'] = beam_results

    # 绘制束宽-性能曲线
    bleu_scores = [beam_results[str(bw)]['bleu_4'] for bw in beam_widths]
    decode_times = [beam_results[str(bw)]['decode_time'] for bw in beam_widths]
    plot_beam_width_performance(
        beam_widths, bleu_scores, decode_times,
        str(img_dir / 'beam_width_performance.png')
    )

    # 3. 温度采样
    print("\n评估温度采样...")
    sampling_results = {}
    temperatures = [0.5, 1.0, 1.5]
    for temp in temperatures:
        print(f"  温度={temp}")
        results = evaluate_decoding_strategy(
            model, test_loader, preprocessor, DEVICE,
            'sampling', temperature=temp
        )
        sampling_results[str(temp)] = results

    all_results['sampling'] = sampling_results

    # 绘制多样性vs质量图
    diversity_scores = [sampling_results[str(t)].get('distinct_2', 0) for t in temperatures]
    quality_scores = [sampling_results[str(t)].get('bleu_4', 0) for t in temperatures]
    labels = [f'T={t}' for t in temperatures]
    plot_diversity_vs_quality(
        diversity_scores, quality_scores, labels,
        str(img_dir / 'diversity_vs_quality.png')
    )

    # 生成概率分布可视化（取一个样本）
    visualize_probability_distribution(
        model, test_loader, preprocessor, DEVICE, str(img_dir)
    )

    # 生成对比图
    generate_comparison_plots(all_results, str(img_dir))

    # 保存结果
    with open(output_dir / 'all_results.json', 'w', encoding='utf-8') as f:
        # 移除不能序列化的内容
        save_results = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                if 'predictions' in v:
                    save_results[k] = {kk: vv for kk, vv in v.items()
                                       if kk not in ['predictions', 'references']}
                else:
                    save_results[k] = {}
                    for kk, vv in v.items():
                        if isinstance(vv, dict):
                            save_results[k][kk] = {kkk: vvv for kkk, vvv in vv.items()
                                                   if kkk not in ['predictions', 'references']}
                        else:
                            save_results[k][kk] = vv
            else:
                save_results[k] = v

        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n实验完成！结果保存在: {output_dir}")


def visualize_probability_distribution(model, test_loader, preprocessor, device, save_dir):
    """可视化解码概率分布"""
    model.eval()

    # 获取一个样本
    batch = next(iter(test_loader))
    src = batch['src'][:1].to(device)
    tgt = batch['tgt'][:1].to(device)

    # 获取前几个时间步的概率分布
    for step in range(min(5, tgt.size(1) - 1)):
        tgt_prefix = tgt[:, :step + 1]

        probs = get_decoding_probability_distribution(
            model, src, tgt_prefix, device
        )

        # 获取词汇表标签
        token_labels = [preprocessor.tgt_vocab.idx2word.get(i, f'<{i}>')
                        for i in range(len(preprocessor.tgt_vocab))]

        plot_decoding_probability(
            probs[0].cpu().numpy(),
            token_labels,
            os.path.join(save_dir, f'prob_dist_step_{step}.png'),
            top_k=10,
            title=f'Decoding Probability at Step {step + 1}'
        )


def generate_comparison_plots(all_results: dict, save_dir: str):
    """生成总体对比图"""
    strategies = ['greedy']
    bleu_scores = [all_results['greedy']['bleu_4']]
    decode_speeds = [all_results['greedy']['tokens_per_second']]

    # 添加最佳束搜索结果
    best_beam = max(all_results['beam_search'].items(),
                    key=lambda x: x[1]['bleu_4'])
    strategies.append(f'beam_{best_beam[0]}')
    bleu_scores.append(best_beam[1]['bleu_4'])
    decode_speeds.append(best_beam[1]['tokens_per_second'])

    # 添加温度1.0的采样结果
    if '1.0' in all_results['sampling']:
        strategies.append('sampling_1.0')
        bleu_scores.append(all_results['sampling']['1.0']['bleu_4'])
        decode_speeds.append(all_results['sampling']['1.0']['tokens_per_second'])

    metrics = {
        'BLEU-4': bleu_scores,
        'Speed (tokens/s)': [s / 100 for s in decode_speeds]  # 缩放以便可视化
    }

    plot_metrics_comparison(
        strategies, metrics,
        os.path.join(save_dir, 'strategy_comparison.png'),
        title='Decoding Strategy Comparison'
    )


if __name__ == '__main__':
    run_experiment()
