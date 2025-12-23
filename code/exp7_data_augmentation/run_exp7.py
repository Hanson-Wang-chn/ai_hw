# -*- coding: utf-8 -*-
"""
实验七：数据增强方法对比实验
探究不同数据增强方法对小样本英德翻译任务的性能提升效果
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.manifold import TSNE

from config.base_config import DEVICE, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
from data.dataset import prepare_data, TranslationDataset, create_dataloaders
from data.preprocess import DataPreprocessor
from data.data_augmentation import DataAugmentor, filter_augmented_data
from model.base_transformer import create_transformer_model
from train.trainer import Trainer, create_optimizer_and_scheduler
from train.utils import compute_bleu
from visualization.plot_utils import (
    plot_training_curves,
    plot_feature_tsne,
    plot_metrics_comparison
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_experiment(config: dict, save_dir: str, preprocessor, device):
    """运行单个实验配置"""
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"运行实验: {config['experiment']['name']}")
    print(f"{'='*50}")

    # 加载原始数据
    print("加载数据...")
    train_data, val_data, test_data = preprocessor.load_multi30k()

    # 处理数据
    train_processed = preprocessor.process_dataset(train_data, build_vocab=True)
    val_processed = preprocessor.process_dataset(val_data, build_vocab=False)
    test_processed = preprocessor.process_dataset(test_data, build_vocab=False)

    # 应用数据增强
    aug_config = config['augmentation']
    aug_method = aug_config['method']

    if aug_method != 'none':
        print(f"应用数据增强: {aug_method}")
        augmentor = DataAugmentor(method=aug_method)

        # 将processed数据转换为增强器需要的格式
        train_texts = [{'src_text': item['src_text'], 'tgt_text': item['tgt_text']}
                       for item in train_processed]

        if aug_method == 'back_translation':
            # 回译增强（注意：这需要较长时间）
            augmented = augmentor.augment(train_texts, ratio=aug_config.get('ratio', 1.0))
        elif aug_method == 'sentence_disturb':
            augmented = augmentor.augment(train_texts, ratio=aug_config.get('ratio', 3.0))
        else:
            augmented = train_texts

        # 过滤质量不佳的增强数据
        augmented = filter_augmented_data(augmented, min_bleu=aug_config.get('min_bleu_filter', 0.6))

        # 控制训练集总量一致
        original_size = len(train_processed)
        target_size = original_size * 2  # 增强后总量为原来的2倍

        if len(augmented) > target_size - original_size:
            augmented = random.sample(augmented, target_size - original_size)

        # 重新处理增强数据
        augmented_processed = []
        for item in augmented:
            tokens_src = preprocessor.tokenize(item['src_text'], is_english=True)
            tokens_tgt = preprocessor.tokenize(item['tgt_text'], is_english=False)
            augmented_processed.append({
                'src_tokens': tokens_src,
                'tgt_tokens': tokens_tgt,
                'src_text': item['src_text'],
                'tgt_text': item['tgt_text']
            })

        train_processed = train_processed + augmented_processed
        print(f"增强后训练集大小: {len(train_processed)}")

    # 编码数据
    train_encoded = preprocessor.encode_data(train_processed)
    val_encoded = preprocessor.encode_data(val_processed)
    test_encoded = preprocessor.encode_data(test_processed)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        preprocessor, train_encoded, val_encoded, test_encoded,
        batch_size=config['training']['batch_size']
    )

    # 创建模型
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
        device=device,
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

    # 评估鲁棒性（在添加噪声的测试集上评估）
    robustness_score = evaluate_robustness(
        model, test_processed, preprocessor, device
    )

    # 可视化数据分布
    if aug_method != 'none' and len(train_processed) > 100:
        visualize_data_distribution(
            model, train_encoded, preprocessor, device, img_dir, aug_method
        )

    results = {
        'config': config,
        'test_metrics': history['test_metrics'],
        'robustness_score': robustness_score,
        'train_size': len(train_processed),
        **history['test_metrics']
    }

    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def evaluate_robustness(model, test_data, preprocessor, device, noise_ratio=0.1):
    """评估模型鲁棒性（在添加噪声的数据上）"""
    model.eval()

    # 添加噪声
    noisy_data = []
    for item in test_data:
        tokens = item['src_tokens'].copy()

        # 随机替换一些词
        for i in range(len(tokens)):
            if random.random() < noise_ratio:
                # 简单地用<unk>替换
                tokens[i] = '<unk>'

        noisy_data.append({
            'src_tokens': tokens,
            'tgt_tokens': item['tgt_tokens'],
            'src_text': ' '.join(tokens),
            'tgt_text': item['tgt_text']
        })

    # 编码并评估
    noisy_encoded = preprocessor.encode_data(noisy_data)

    from infer.decoder import greedy_decode

    predictions = []
    references = []

    with torch.no_grad():
        for item in noisy_encoded[:100]:  # 只评估部分样本
            src = torch.tensor([item['src']], device=device)

            pred = greedy_decode(
                model, src,
                preprocessor.tgt_vocab.sos_idx,
                preprocessor.tgt_vocab.eos_idx,
                max_len=DATA_CONFIG['max_seq_len'],
                device=device
            )

            pred_tokens = []
            for idx in pred[0].tolist():
                if idx == preprocessor.tgt_vocab.eos_idx:
                    break
                if idx not in [preprocessor.tgt_vocab.pad_idx, preprocessor.tgt_vocab.sos_idx]:
                    pred_tokens.append(preprocessor.tgt_vocab.idx2word.get(idx, '<unk>'))

            ref_tokens = []
            for idx in item['tgt']:
                if idx == preprocessor.tgt_vocab.eos_idx:
                    break
                if idx not in [preprocessor.tgt_vocab.pad_idx, preprocessor.tgt_vocab.sos_idx]:
                    ref_tokens.append(preprocessor.tgt_vocab.idx2word.get(idx, '<unk>'))

            predictions.append(pred_tokens)
            references.append(ref_tokens)

    # 计算BLEU
    if predictions:
        bleu = compute_bleu(predictions, references)
        return bleu.get('bleu_4', 0)
    return 0


def visualize_data_distribution(model, train_encoded, preprocessor, device, save_dir, aug_method):
    """可视化原始数据与增强数据的分布"""
    model.eval()

    features = []
    labels = []

    # 假设前一半是原始数据，后一半是增强数据
    original_size = len(train_encoded) // 2

    with torch.no_grad():
        for i, item in enumerate(train_encoded[:500]):
            src = torch.tensor([item['src']], device=device)
            src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)

            encoder_output, _ = model.encoder(src, src_mask)

            # 平均池化
            mask = (src != model.pad_idx).unsqueeze(-1).float()
            pooled = (encoder_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

            features.append(pooled.cpu().numpy().flatten())
            labels.append(0 if i < original_size else 1)

    features = np.array(features)
    labels = np.array(labels)

    plot_feature_tsne(
        features, labels,
        os.path.join(save_dir, f'data_distribution_{aug_method}.png'),
        title=f'Original vs Augmented Data Distribution ({aug_method})'
    )


def run_experiment():
    """运行完整的数据增强实验"""
    parser = argparse.ArgumentParser(description='实验七：数据增强方法对比实验')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./exp7_data_augmentation/results')
    args = parser.parse_args()

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

        # 每次创建新的preprocessor以确保词汇表正确
        preprocessor = DataPreprocessor()

        try:
            results = run_single_experiment(
                config, str(save_dir), preprocessor, DEVICE
            )
            all_results[exp_name] = results
        except Exception as e:
            print(f"实验 {exp_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {'error': str(e)}

    if len(all_results) > 1:
        generate_comparison_plots(all_results, str(output_dir / 'comparison'))

    with open(output_dir / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n实验完成！结果保存在: {output_dir}")


def generate_comparison_plots(all_results: dict, save_dir: str):
    """生成对比可视化"""
    os.makedirs(save_dir, exist_ok=True)

    configs = [k for k in all_results.keys() if 'error' not in all_results[k]]
    if not configs:
        return

    metrics = {
        'BLEU-4': [all_results[c].get('bleu_4', 0) for c in configs],
        'Robustness': [all_results[c].get('robustness_score', 0) for c in configs]
    }

    plot_metrics_comparison(
        configs, metrics,
        os.path.join(save_dir, 'metrics_comparison.png'),
        title='Data Augmentation Methods Comparison'
    )


if __name__ == '__main__':
    run_experiment()
