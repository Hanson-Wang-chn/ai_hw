# -*- coding: utf-8 -*-
"""
实验六：预训练与微调策略对比实验
探究预训练+微调策略在小样本英德翻译任务中的有效性
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
from torch.utils.data import DataLoader

from config.base_config import DEVICE, MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
from data.dataset import prepare_data
from data.preprocess import DataPreprocessor
from model.base_transformer import create_transformer_model
from model.pretrain_model import get_pretrained_model
from train.trainer import Trainer, create_optimizer_and_scheduler
from train.finetune_trainer import FinetuneTrainer, PretrainDataset
from train.utils import compute_bleu
from visualization.plot_utils import (
    plot_training_curves,
    plot_metrics_comparison,
    plot_layer_gradient_heatmap
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_no_pretrain_experiment(config, data_loaders, preprocessor, save_dir):
    """运行无预训练实验"""
    train_loader, val_loader, test_loader = data_loaders

    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

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

    history = trainer.train(train_config['epochs'])

    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        os.path.join(img_dir, 'training_curves.png'),
        val_metric=history['val_bleu_4'],
        metric_name='BLEU-4'
    )

    return {
        'test_metrics': history['test_metrics'],
        'convergence_epoch': find_convergence_epoch(history['val_bleu_4']),
        'train_loss_history': history['train_loss'],
        **history['test_metrics']
    }


def run_pretrain_experiment(config, save_dir):
    """运行预训练微调实验"""
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    pretrain_config = config['pretrain']
    train_config = config['training']

    # 加载预训练模型
    print(f"加载预训练模型: {pretrain_config['model_type']}")
    model = get_pretrained_model(
        model_type=pretrain_config['model_type'],
        src_lang='en',
        tgt_lang='de',
        adapter_dim=pretrain_config.get('adapter_dim', 64),
        use_adapter=(pretrain_config['finetune_strategy'] == 'adapter')
    )

    # 应用微调策略
    if pretrain_config['finetune_strategy'] == 'full':
        model.unfreeze_parameters()
    elif pretrain_config['finetune_strategy'] == 'layer_freeze':
        model.freeze_layers(
            encoder_layers=pretrain_config.get('encoder_freeze_layers', 3),
            decoder_layers=pretrain_config.get('decoder_freeze_layers', 2)
        )
    elif pretrain_config['finetune_strategy'] == 'adapter':
        model.freeze_parameters()
        # 适配器参数在模型创建时已经设置为可训练

    model = model.to(DEVICE)

    print(f"可训练参数量: {model.count_trainable_parameters():,}")

    # 加载数据
    print("加载Multi30K数据...")
    preprocessor = DataPreprocessor()
    train_data, val_data, test_data = preprocessor.load_multi30k()

    # 处理数据为预训练模型格式
    train_processed = preprocessor.process_dataset(train_data, build_vocab=False)
    val_processed = preprocessor.process_dataset(val_data, build_vocab=False)
    test_processed = preprocessor.process_dataset(test_data, build_vocab=False)

    # 创建数据集和加载器
    train_dataset = PretrainDataset(
        train_processed, model.tokenizer,
        max_length=config['data'].get('max_seq_len', 128)
    )
    val_dataset = PretrainDataset(
        val_processed, model.tokenizer,
        max_length=config['data'].get('max_seq_len', 128)
    )
    test_dataset = PretrainDataset(
        test_processed, model.tokenizer,
        max_length=config['data'].get('max_seq_len', 128)
    )

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'])

    # 创建微调训练器
    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_config,
        save_dir=save_dir,
        device=DEVICE,
        finetune_strategy=pretrain_config['finetune_strategy']
    )

    # 训练
    history = trainer.train(train_config['epochs'])

    # 绘制训练曲线
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        os.path.join(img_dir, 'training_curves.png'),
        val_metric=history.get('val_bleu_4', []),
        metric_name='BLEU-4'
    )

    return {
        'test_metrics': history.get('test_metrics', {}),
        'convergence_epoch': find_convergence_epoch(history.get('val_bleu_4', [])),
        'trainable_params': model.count_trainable_parameters(),
        **history.get('test_metrics', {})
    }


def find_convergence_epoch(bleu_history, threshold=0.8):
    """找到达到最佳BLEU 80%的epoch"""
    if not bleu_history:
        return -1
    max_bleu = max(bleu_history) if bleu_history else 0
    if max_bleu == 0:
        return -1
    target = max_bleu * threshold
    for epoch, bleu in enumerate(bleu_history, 1):
        if bleu >= target:
            return epoch
    return len(bleu_history)


def run_experiment():
    """运行完整的预训练微调实验"""
    parser = argparse.ArgumentParser(description='实验六：预训练与微调策略对比实验')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./exp6_pretrain/results')
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

        print(f"\n{'='*50}")
        print(f"运行实验: {exp_name}")
        print(f"{'='*50}")

        try:
            if config['pretrain']['model_type'] == 'none':
                # 无预训练实验
                print("准备数据...")
                train_loader, val_loader, test_loader, preprocessor = prepare_data(
                    batch_size=config['training']['batch_size'],
                    data_dir='./data_cache'
                )
                results = run_no_pretrain_experiment(
                    config, (train_loader, val_loader, test_loader),
                    preprocessor, str(save_dir)
                )
            else:
                # 预训练微调实验
                results = run_pretrain_experiment(config, str(save_dir))

            all_results[exp_name] = results

            with open(save_dir / 'results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            print(f"实验 {exp_name} 失败: {e}")
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
        'Convergence Epoch': [all_results[c].get('convergence_epoch', 0) for c in configs]
    }

    plot_metrics_comparison(
        configs, metrics,
        os.path.join(save_dir, 'metrics_comparison.png'),
        title='Pretrain & Finetune Strategy Comparison'
    )


if __name__ == '__main__':
    run_experiment()
