# -*- coding: utf-8 -*-
"""
Seq2Seq模型实验统一运行入口
使用方法: python run.py --task <task_name> [--config <config_file>]

支持的任务:
- exp1_position_encoding: 位置编码对比实验
- exp2_attention: 注意力机制对比实验
- exp3_feedforward: FeedForward网络对比实验
- exp4_normalization: 归一化策略对比实验
- exp5_decoding: 解码策略优化实验
- exp6_pretrain: 预训练与微调策略实验
- exp7_data_augmentation: 数据增强方法实验
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description='Seq2Seq模型实验统一运行入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --task exp1_position_encoding
  python run.py --task exp2_attention --config multi_head.yaml
  python run.py --task exp5_decoding --model_path ./exp1_position_encoding/results/best_model.pth
        """
    )

    parser.add_argument(
        '--task', type=str, required=True,
        choices=[
            'exp1_position_encoding',
            'exp2_attention',
            'exp3_feedforward',
            'exp4_normalization',
            'exp5_decoding',
            'exp6_pretrain',
            'exp7_data_augmentation'
        ],
        help='要运行的实验任务名称'
    )

    parser.add_argument(
        '--config', type=str, default=None,
        help='指定单个配置文件（可选，默认运行该实验的所有配置）'
    )

    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='输出目录（可选，默认为 ./<task>/results）'
    )

    parser.add_argument(
        '--model_path', type=str, default=None,
        help='预训练模型路径（仅用于exp5_decoding）'
    )

    args = parser.parse_args()

    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = str(ROOT_DIR / args.task / 'results')

    # 根据任务导入并运行相应的实验脚本
    print(f"{'='*60}")
    print(f"运行实验: {args.task}")
    print(f"输出目录: {args.output_dir}")
    if args.config:
        print(f"配置文件: {args.config}")
    print(f"{'='*60}")

    # 动态导入并运行实验
    if args.task == 'exp1_position_encoding':
        from exp1_position_encoding.run_exp1 import run_experiment
        # 构建命令行参数
        sys.argv = ['run_exp1.py', '--output_dir', args.output_dir]
        if args.config:
            sys.argv.extend(['--config', args.config])
        run_experiment()

    elif args.task == 'exp2_attention':
        from exp2_attention.run_exp2 import run_experiment
        sys.argv = ['run_exp2.py', '--output_dir', args.output_dir]
        if args.config:
            sys.argv.extend(['--config', args.config])
        run_experiment()

    elif args.task == 'exp3_feedforward':
        from exp3_feedforward.run_exp3 import run_experiment
        sys.argv = ['run_exp3.py', '--output_dir', args.output_dir]
        if args.config:
            sys.argv.extend(['--config', args.config])
        run_experiment()

    elif args.task == 'exp4_normalization':
        from exp4_normalization.run_exp4 import run_experiment
        sys.argv = ['run_exp4.py', '--output_dir', args.output_dir]
        if args.config:
            sys.argv.extend(['--config', args.config])
        run_experiment()

    elif args.task == 'exp5_decoding':
        from exp5_decoding.run_exp5 import run_experiment
        sys.argv = ['run_exp5.py', '--output_dir', args.output_dir]
        if args.model_path:
            sys.argv.extend(['--model_path', args.model_path])
        run_experiment()

    elif args.task == 'exp6_pretrain':
        from exp6_pretrain.run_exp6 import run_experiment
        sys.argv = ['run_exp6.py', '--output_dir', args.output_dir]
        if args.config:
            sys.argv.extend(['--config', args.config])
        run_experiment()

    elif args.task == 'exp7_data_augmentation':
        from exp7_data_augmentation.run_exp7 import run_experiment
        sys.argv = ['run_exp7.py', '--output_dir', args.output_dir]
        if args.config:
            sys.argv.extend(['--config', args.config])
        run_experiment()

    else:
        print(f"未知的任务: {args.task}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"实验完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
