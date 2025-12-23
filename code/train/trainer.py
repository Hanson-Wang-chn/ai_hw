# -*- coding: utf-8 -*-
"""
通用训练器
包含训练循环、验证、评估等功能
"""

import os
import time
from typing import Dict, Optional, Callable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from train.utils import (
    compute_loss,
    compute_all_metrics,
    compute_gradient_norm,
    check_gradient_health,
    setup_logger,
    MetricsTracker
)
from infer.decoder import greedy_decode


class Trainer:
    """通用训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        config: Dict,
        save_dir: str,
        device: torch.device,
        preprocessor=None
    ):
        """
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 训练配置
            save_dir: 保存目录
            device: 计算设备
            preprocessor: 数据预处理器（用于解码）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.save_dir = save_dir
        self.device = device
        self.preprocessor = preprocessor

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

        # 设置日志
        self.logger = setup_logger(os.path.join(save_dir, 'logs'))
        self.writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

        # 指标跟踪
        self.metrics_tracker = MetricsTracker()

        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # 早停
        self.patience = config.get('patience', 5)
        self.patience_counter = 0
        self.best_metric = 0.0

        # 梯度裁剪
        self.grad_clip = config.get('grad_clip', 1.0)

        # 标签平滑
        self.label_smoothing = config.get('label_smoothing', 0.1)

        # 获取pad_idx
        self.pad_idx = getattr(preprocessor.src_vocab, 'pad_idx', 0) if preprocessor else 0

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_bleu_4': [],
            'learning_rate': [],
            'gradient_norms': []
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_grad_norms = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')

        for batch in pbar:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            self.optimizer.zero_grad()

            # 混合精度前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(src, tgt)
                    loss = compute_loss(
                        outputs['output'], tgt,
                        self.pad_idx, self.label_smoothing
                    )

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(src, tgt)
                loss = compute_loss(
                    outputs['output'], tgt,
                    self.pad_idx, self.label_smoothing
                )

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

            # 记录梯度范数
            grad_norms = compute_gradient_norm(self.model)
            all_grad_norms.append(grad_norms)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        # 平均梯度范数
        avg_grad_norm = {}
        for key in all_grad_norms[0].keys():
            avg_grad_norm[key] = sum(g[key] for g in all_grad_norms) / len(all_grad_norms)

        return {
            'train_loss': avg_loss,
            **avg_grad_norm
        }

    @torch.no_grad()
    def validate(self, data_loader: DataLoader, desc: str = 'Val') -> Dict[str, float]:
        """
        验证/测试

        Args:
            data_loader: 数据加载器
            desc: 描述

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_hypotheses = []
        all_references = []
        all_hyp_tokens = []
        all_ref_tokens = []

        pbar = tqdm(data_loader, desc=desc)

        for batch in pbar:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # 前向传播计算损失
            outputs = self.model(src, tgt)
            loss = compute_loss(outputs['output'], tgt, self.pad_idx)
            total_loss += loss.item()
            num_batches += 1

            # 生成预测
            predictions = greedy_decode(
                self.model, src,
                self.preprocessor.tgt_vocab.sos_idx,
                self.preprocessor.tgt_vocab.eos_idx,
                max_len=self.config.get('max_seq_len', 20),
                device=self.device
            )

            # 解码预测和参考
            for pred, ref in zip(predictions, tgt):
                # 移除特殊token
                pred_tokens = []
                for idx in pred.tolist():
                    if idx == self.preprocessor.tgt_vocab.eos_idx:
                        break
                    if idx not in [self.preprocessor.tgt_vocab.pad_idx,
                                   self.preprocessor.tgt_vocab.sos_idx]:
                        pred_tokens.append(self.preprocessor.tgt_vocab.idx2word.get(idx, '<unk>'))

                ref_tokens = []
                for idx in ref.tolist():
                    if idx == self.preprocessor.tgt_vocab.eos_idx:
                        break
                    if idx not in [self.preprocessor.tgt_vocab.pad_idx,
                                   self.preprocessor.tgt_vocab.sos_idx]:
                        ref_tokens.append(self.preprocessor.tgt_vocab.idx2word.get(idx, '<unk>'))

                all_hyp_tokens.append(pred_tokens)
                all_ref_tokens.append(ref_tokens)
                all_hypotheses.append(' '.join(pred_tokens))
                all_references.append(' '.join(ref_tokens))

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        # 计算所有指标
        metrics = compute_all_metrics(
            all_hypotheses, all_references,
            all_hyp_tokens, all_ref_tokens
        )
        metrics['loss'] = avg_loss

        return metrics

    def train(self, num_epochs: int) -> Dict:
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数

        Returns:
            训练历史
        """
        self.logger.info(f"开始训练，共 {num_epochs} 轮")
        self.logger.info(f"模型参数量: {self.model.count_parameters():,}")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['gradient_norms'].append(train_metrics.get('grad_norm_total', 0))

            # 验证
            val_metrics = self.validate(self.val_loader, f'Epoch {epoch} [Val]')
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_bleu_4'].append(val_metrics.get('bleu_4', 0))

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # 日志记录
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_bleu_4={val_metrics.get('bleu_4', 0):.4f}, "
                f"lr={current_lr:.2e}"
            )

            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('BLEU-4/val', val_metrics.get('bleu_4', 0), epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)

            for key, value in train_metrics.items():
                if 'grad_norm' in key:
                    self.writer.add_scalar(f'Gradient/{key}', value, epoch)

            # 更新指标跟踪器
            self.metrics_tracker.update({**train_metrics, **val_metrics}, epoch)

            # 学习率调度
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('bleu_4', 0))
            else:
                self.scheduler.step()

            # 保存最佳模型
            current_metric = val_metrics.get('bleu_4', 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
                self.logger.info(f"新的最佳模型！BLEU-4: {current_metric:.4f}")
            else:
                self.patience_counter += 1

            # 早停检查
            if self.patience_counter >= self.patience:
                self.logger.info(f"早停触发，已有 {self.patience} 轮无提升")
                break

            # 检查梯度健康
            health = check_gradient_health(train_metrics)
            if not health['is_healthy']:
                self.logger.warning(f"梯度异常: {health}")

        # 训练结束
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，耗时: {total_time / 60:.2f} 分钟")

        # 测试集评估
        self.logger.info("在测试集上评估最佳模型...")
        self.load_checkpoint(is_best=True)
        test_metrics = self.validate(self.test_loader, 'Test')
        self.logger.info(f"测试集结果: {test_metrics}")

        self.history['test_metrics'] = test_metrics

        self.writer.close()

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        if is_best:
            path = os.path.join(self.save_dir, 'checkpoints', 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """加载检查点"""
        if is_best:
            path = os.path.join(self.save_dir, 'checkpoints', 'best_model.pth')

        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_metric = checkpoint.get('best_metric', 0)
            self.logger.info(f"加载检查点: {path}")


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict
) -> tuple:
    """
    创建优化器和学习率调度器

    Args:
        model: PyTorch模型
        config: 配置字典

    Returns:
        (optimizer, scheduler)
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        betas=config.get('betas', (0.9, 0.98)),
        weight_decay=config.get('weight_decay', 0.01)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.get('lr_factor', 0.5),
        patience=config.get('lr_patience', 3),
        verbose=True
    )

    return optimizer, scheduler
