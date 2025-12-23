# -*- coding: utf-8 -*-
"""
推理脚本
支持单样本和批量推理
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import time

from infer.decoder import Decoder, greedy_decode, beam_search_decode, temperature_sampling_decode


class Translator:
    """翻译器"""

    def __init__(
        self,
        model: nn.Module,
        preprocessor,
        device: torch.device,
        max_len: int = 50
    ):
        """
        Args:
            model: 训练好的模型
            preprocessor: 数据预处理器
            device: 计算设备
            max_len: 最大生成长度
        """
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.device = device
        self.max_len = max_len

        # 初始化解码器
        self.decoder = Decoder(
            model,
            preprocessor.tgt_vocab.sos_idx,
            preprocessor.tgt_vocab.eos_idx,
            device
        )

    def preprocess(self, text: str) -> torch.Tensor:
        """预处理输入文本"""
        # 清洗和分词
        clean_text = self.preprocessor.clean_text(text, is_english=True)
        tokens = self.preprocessor.tokenize(clean_text, is_english=True)

        # 编码
        indices = self.preprocessor.src_vocab.encode(tokens)
        indices = [self.preprocessor.src_vocab.sos_idx] + indices + [self.preprocessor.src_vocab.eos_idx]

        # 截断
        indices = indices[:self.max_len]

        return torch.tensor([indices], dtype=torch.long, device=self.device)

    def postprocess(self, indices: List[int]) -> str:
        """后处理输出索引"""
        tokens = []
        for idx in indices:
            if idx == self.preprocessor.tgt_vocab.eos_idx:
                break
            if idx not in [
                self.preprocessor.tgt_vocab.pad_idx,
                self.preprocessor.tgt_vocab.sos_idx
            ]:
                token = self.preprocessor.tgt_vocab.idx2word.get(idx, '<unk>')
                tokens.append(token)
        return ' '.join(tokens)

    def translate(
        self,
        text: str,
        strategy: str = 'greedy',
        **kwargs
    ) -> Dict:
        """
        翻译单个文本

        Args:
            text: 输入文本
            strategy: 解码策略
            **kwargs: 策略参数

        Returns:
            包含翻译结果和相关信息的字典
        """
        start_time = time.time()

        # 预处理
        src = self.preprocess(text)

        # 解码
        result = self.decoder.decode(src, strategy, max_len=self.max_len, **kwargs)

        # 后处理
        sequences = result['sequences']
        translation = self.postprocess(sequences[0].tolist())

        elapsed_time = time.time() - start_time

        return {
            'input': text,
            'translation': translation,
            'strategy': strategy,
            'time': elapsed_time,
            **{k: v for k, v in result.items() if k != 'sequences'}
        }

    def translate_batch(
        self,
        texts: List[str],
        strategy: str = 'greedy',
        **kwargs
    ) -> List[Dict]:
        """
        批量翻译

        Args:
            texts: 输入文本列表
            strategy: 解码策略
            **kwargs: 策略参数

        Returns:
            翻译结果列表
        """
        results = []
        for text in texts:
            result = self.translate(text, strategy, **kwargs)
            results.append(result)
        return results

    def interactive_translate(self):
        """交互式翻译"""
        print("=" * 50)
        print("交互式翻译模式")
        print("输入 'quit' 退出")
        print("输入 'strategy:xxx' 切换解码策略 (greedy/beam/sampling)")
        print("=" * 50)

        strategy = 'greedy'
        strategy_params = {}

        while True:
            try:
                text = input("\n请输入英文: ").strip()

                if text.lower() == 'quit':
                    break

                if text.startswith('strategy:'):
                    strategy = text.split(':')[1].strip()
                    if strategy == 'beam':
                        strategy_params = {'beam_width': 4}
                    elif strategy == 'sampling':
                        strategy_params = {'temperature': 1.0}
                    else:
                        strategy_params = {}
                    print(f"已切换到 {strategy} 策略")
                    continue

                result = self.translate(text, strategy, **strategy_params)
                print(f"德语翻译: {result['translation']}")
                print(f"耗时: {result['time']:.3f}秒")

            except KeyboardInterrupt:
                print("\n退出翻译模式")
                break

    def compute_speed(
        self,
        texts: List[str],
        strategy: str = 'greedy',
        **kwargs
    ) -> Dict[str, float]:
        """
        计算翻译速度

        Args:
            texts: 测试文本列表
            strategy: 解码策略
            **kwargs: 策略参数

        Returns:
            速度统计
        """
        total_tokens = 0
        total_time = 0.0

        for text in texts:
            start_time = time.time()

            src = self.preprocess(text)
            result = self.decoder.decode(src, strategy, max_len=self.max_len, **kwargs)
            sequences = result['sequences']

            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # 计算生成的token数
            gen_len = sequences.size(1)
            total_tokens += gen_len

        return {
            'total_tokens': total_tokens,
            'total_time': total_time,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'sentences_per_second': len(texts) / total_time if total_time > 0 else 0
        }


def load_model_for_inference(
    checkpoint_path: str,
    preprocessor,
    model_config: Dict,
    device: torch.device
) -> Translator:
    """
    加载模型用于推理

    Args:
        checkpoint_path: 检查点路径
        preprocessor: 数据预处理器
        model_config: 模型配置
        device: 计算设备

    Returns:
        Translator对象
    """
    from model.base_transformer import create_transformer_model

    # 创建模型
    model = create_transformer_model(
        len(preprocessor.src_vocab),
        len(preprocessor.tgt_vocab),
        model_config,
        preprocessor.src_vocab.pad_idx
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已从 {checkpoint_path} 加载")

    return Translator(model, preprocessor, device)
