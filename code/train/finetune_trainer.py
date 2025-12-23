# -*- coding: utf-8 -*-
"""
预训练模型微调训练器
包含层冻结、学习率衰减、适配器训练等策略
"""

import os
import time
from typing import Dict, Optional, List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train.utils import setup_logger, MetricsTracker
from train.trainer import Trainer


class FinetuneTrainer(Trainer):
    """预训练模型微调训练器"""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        save_dir: str,
        device: torch.device,
        finetune_strategy: str = 'full'
    ):
        """
        Args:
            model: 预训练模型包装器
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            config: 训练配置
            save_dir: 保存目录
            device: 计算设备
            finetune_strategy: 微调策略
                - 'full': 全参数微调
                - 'layer_freeze': 分层冻结
                - 'adapter': 适配器微调
        """
        self.finetune_strategy = finetune_strategy
        self.pretrained_model = model

        # 将预训练模型移至设备
        self.pretrained_model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.save_dir = save_dir
        self.device = device

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

        # 设置日志
        self.logger = setup_logger(os.path.join(save_dir, 'logs'))
        self.writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

        # 指标跟踪
        self.metrics_tracker = MetricsTracker()

        # 应用微调策略
        self._apply_finetune_strategy()

        # 创建优化器（针对不同策略）
        self.optimizer = self._create_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 3)
        )

        # 早停
        self.patience = config.get('patience', 5)
        self.patience_counter = 0
        self.best_metric = 0.0

        # 梯度裁剪
        self.grad_clip = config.get('grad_clip', 1.0)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_bleu_4': [],
            'learning_rate': [],
            'trainable_params': self.pretrained_model.count_trainable_parameters()
        }

        self.logger.info(f"微调策略: {finetune_strategy}")
        self.logger.info(f"可训练参数量: {self.history['trainable_params']:,}")

    def _apply_finetune_strategy(self):
        """应用微调策略"""
        if self.finetune_strategy == 'full':
            # 全参数微调
            self.pretrained_model.unfreeze_parameters()

        elif self.finetune_strategy == 'layer_freeze':
            # 分层冻结
            encoder_freeze = self.config.get('encoder_freeze_layers', 3)
            decoder_freeze = self.config.get('decoder_freeze_layers', 2)
            self.pretrained_model.freeze_layers(encoder_freeze, decoder_freeze)

        elif self.finetune_strategy == 'adapter':
            # 适配器微调：冻结所有预训练参数，仅训练适配器
            self.pretrained_model.freeze_parameters()
            # 解冻适配器参数
            if hasattr(self.pretrained_model, 'adapters'):
                for adapter in self.pretrained_model.adapters.values():
                    for param in adapter.parameters():
                        param.requires_grad = True

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器（针对不同策略使用不同学习率）"""
        base_lr = self.config.get('lr', 1e-4)
        finetune_lr = self.config.get('finetune_lr', 1e-5)

        if self.finetune_strategy == 'full':
            # 全参数微调使用较小学习率
            return optim.AdamW(
                filter(lambda p: p.requires_grad, self.pretrained_model.parameters()),
                lr=finetune_lr,
                weight_decay=self.config.get('weight_decay', 0.01)
            )

        elif self.finetune_strategy == 'layer_freeze':
            # 分层学习率：冻结层不更新，其他层使用微调学习率
            return optim.AdamW(
                filter(lambda p: p.requires_grad, self.pretrained_model.parameters()),
                lr=finetune_lr,
                weight_decay=self.config.get('weight_decay', 0.01)
            )

        elif self.finetune_strategy == 'adapter':
            # 适配器使用较大学习率
            return optim.AdamW(
                filter(lambda p: p.requires_grad, self.pretrained_model.parameters()),
                lr=base_lr,
                weight_decay=self.config.get('weight_decay', 0.01)
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.pretrained_model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.pretrained_model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return {'train_loss': total_loss / num_batches}

    @torch.no_grad()
    def validate(self, data_loader: DataLoader, desc: str = 'Val') -> Dict[str, float]:
        """验证"""
        self.pretrained_model.eval()
        total_loss = 0.0
        num_batches = 0

        all_hypotheses = []
        all_references = []

        pbar = tqdm(data_loader, desc=desc)

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            references = batch.get('reference_texts', [])

            # 计算损失
            outputs = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            num_batches += 1

            # 生成预测
            generated = self.pretrained_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.get('max_seq_len', 50),
                num_beams=4
            )

            # 解码
            decoded = self.pretrained_model.decode_batch(generated)
            all_hypotheses.extend(decoded)
            all_references.extend(references)

            pbar.set_postfix({'loss': f'{outputs.loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        # 计算BLEU（简化版本）
        from train.utils import compute_bleu
        tokenized_hyp = [h.split() for h in all_hypotheses]
        tokenized_ref = [r.split() for r in all_references]

        if tokenized_hyp and tokenized_ref:
            bleu = compute_bleu(tokenized_hyp, tokenized_ref)
        else:
            bleu = {'bleu_1': 0, 'bleu_2': 0, 'bleu_4': 0}

        return {
            'loss': avg_loss,
            **bleu
        }

    def compute_pretrain_knowledge_retention(self) -> float:
        """
        计算预训练知识保留率
        通过在通用翻译任务上评估来衡量
        """
        # 简化实现：使用验证集损失作为间接指标
        # 实际应用中应使用预训练任务的测试集
        return self.best_metric

    def get_layer_gradient_norms(self) -> Dict[str, List[float]]:
        """获取各层的梯度范数"""
        layer_norms = {'encoder': [], 'decoder': []}

        if hasattr(self.pretrained_model, 'model'):
            # 编码器层
            if hasattr(self.pretrained_model.model.model, 'encoder'):
                for i, layer in enumerate(self.pretrained_model.model.model.encoder.layers):
                    norm = 0.0
                    count = 0
                    for param in layer.parameters():
                        if param.grad is not None:
                            norm += param.grad.data.norm(2).item()
                            count += 1
                    if count > 0:
                        layer_norms['encoder'].append(norm / count)

            # 解码器层
            if hasattr(self.pretrained_model.model.model, 'decoder'):
                for i, layer in enumerate(self.pretrained_model.model.model.decoder.layers):
                    norm = 0.0
                    count = 0
                    for param in layer.parameters():
                        if param.grad is not None:
                            norm += param.grad.data.norm(2).item()
                            count += 1
                    if count > 0:
                        layer_norms['decoder'].append(norm / count)

        return layer_norms


class PretrainDataset(torch.utils.data.Dataset):
    """预训练模型微调数据集"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 编码源文本
        src_encoding = self.tokenizer(
            item['src_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码目标文本
        with self.tokenizer.as_target_tokenizer():
            tgt_encoding = self.tokenizer(
                item['tgt_text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': tgt_encoding['input_ids'].squeeze(),
            'reference_texts': item['tgt_text']
        }
