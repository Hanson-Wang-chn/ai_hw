# -*- coding: utf-8 -*-
"""
自定义Dataset类
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

from data.preprocess import DataPreprocessor, collate_fn


class TranslationDataset(Dataset):
    """翻译任务数据集"""

    def __init__(self, encoded_data: List[Dict]):
        """
        Args:
            encoded_data: 编码后的数据列表，每个元素包含src, tgt等字段
        """
        self.data = encoded_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def create_dataloaders(
    preprocessor: DataPreprocessor,
    train_encoded: List[Dict],
    val_encoded: List[Dict],
    test_encoded: List[Dict],
    batch_size: int = 32,
    num_workers: int = 0
) -> tuple:
    """
    创建数据加载器

    Args:
        preprocessor: 数据预处理器，包含词汇表
        train_encoded: 训练集编码数据
        val_encoded: 验证集编码数据
        test_encoded: 测试集编码数据
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = TranslationDataset(train_encoded)
    val_dataset = TranslationDataset(val_encoded)
    test_dataset = TranslationDataset(test_encoded)

    # 获取填充索引
    pad_idx = preprocessor.src_vocab.pad_idx

    # 创建collate函数
    collate = partial(collate_fn, pad_idx=pad_idx)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def prepare_data(
    batch_size: int = 32,
    data_dir: str = './data_cache',
    rebuild_vocab: bool = False
) -> tuple:
    """
    准备数据的便捷函数

    Args:
        batch_size: 批次大小
        data_dir: 数据缓存目录
        rebuild_vocab: 是否重新构建词汇表

    Returns:
        train_loader, val_loader, test_loader, preprocessor
    """
    import os
    import pickle

    preprocessor = DataPreprocessor()
    vocab_path = os.path.join(data_dir, 'vocab')
    cache_path = os.path.join(data_dir, 'processed_data.pkl')

    # 尝试加载缓存
    if os.path.exists(cache_path) and os.path.exists(vocab_path) and not rebuild_vocab:
        print("加载缓存数据...")
        preprocessor.load_vocab(vocab_path)
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        train_encoded = cache['train']
        val_encoded = cache['val']
        test_encoded = cache['test']
    else:
        print("处理数据集...")
        os.makedirs(data_dir, exist_ok=True)

        # 加载原始数据
        train_data, val_data, test_data = preprocessor.load_multi30k()

        # 处理数据
        train_processed = preprocessor.process_dataset(train_data, build_vocab=True)
        val_processed = preprocessor.process_dataset(val_data, build_vocab=False)
        test_processed = preprocessor.process_dataset(test_data, build_vocab=False)

        # 编码数据
        train_encoded = preprocessor.encode_data(train_processed)
        val_encoded = preprocessor.encode_data(val_processed)
        test_encoded = preprocessor.encode_data(test_processed)

        # 保存词汇表和缓存
        preprocessor.save_vocab(vocab_path)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'train': train_encoded,
                'val': val_encoded,
                'test': test_encoded
            }, f)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        preprocessor, train_encoded, val_encoded, test_encoded,
        batch_size=batch_size
    )

    print(f"训练集: {len(train_encoded)} 样本")
    print(f"验证集: {len(val_encoded)} 样本")
    print(f"测试集: {len(test_encoded)} 样本")

    return train_loader, val_loader, test_loader, preprocessor
