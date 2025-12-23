# -*- coding: utf-8 -*-
"""
预训练模型封装
包含mBART、M2M-100等预训练模型的加载和适配器实现
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)


class Adapter(nn.Module):
    """
    适配器层
    在预训练模型的每层中添加小型可训练模块
    """

    def __init__(self, hidden_size: int, adapter_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 初始化为接近恒等映射
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, hidden_size)

        Returns:
            输出张量（与输入形状相同）
        """
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class PretrainedModelWrapper(nn.Module):
    """预训练模型包装器基类"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

    def freeze_parameters(self):
        """冻结所有预训练参数"""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_layers(self, encoder_layers: int = 3, decoder_layers: int = 2):
        """
        分层冻结策略

        Args:
            encoder_layers: 冻结编码器前n层
            decoder_layers: 冻结解码器前n层
        """
        raise NotImplementedError

    def count_trainable_parameters(self) -> int:
        """计算可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MBartWrapper(PretrainedModelWrapper):
    """mBART模型包装器"""

    def __init__(
        self,
        model_name: str = 'facebook/mbart-large-50-many-to-many-mmt',
        src_lang: str = 'en_XX',
        tgt_lang: str = 'de_DE',
        adapter_dim: int = 64,
        use_adapter: bool = False
    ):
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_adapter = use_adapter
        self.adapter_dim = adapter_dim

        # 加载预训练模型和分词器
        print(f"加载mBART模型: {model_name}")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)

        # 设置源语言
        self.tokenizer.src_lang = src_lang

        # 如果使用适配器，添加适配器层
        if use_adapter:
            self._add_adapters()

    def _add_adapters(self):
        """在模型中添加适配器层"""
        hidden_size = self.model.config.d_model
        self.adapters = nn.ModuleDict()

        # 为编码器每层添加适配器
        for i in range(len(self.model.model.encoder.layers)):
            self.adapters[f'encoder_{i}'] = Adapter(hidden_size, self.adapter_dim)

        # 为解码器每层添加适配器
        for i in range(len(self.model.model.decoder.layers)):
            self.adapters[f'decoder_{i}'] = Adapter(hidden_size, self.adapter_dim)

    def freeze_layers(self, encoder_layers: int = 3, decoder_layers: int = 2):
        """分层冻结策略"""
        # 冻结词嵌入
        for param in self.model.model.shared.parameters():
            param.requires_grad = False

        # 冻结编码器前n层
        for i, layer in enumerate(self.model.model.encoder.layers):
            if i < encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # 冻结解码器前n层
        for i, layer in enumerate(self.model.model.decoder.layers):
            if i < decoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=return_dict
        )
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        num_beams: int = 4,
        **kwargs
    ) -> torch.Tensor:
        """生成翻译"""
        forced_bos_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            forced_bos_token_id=forced_bos_token_id,
            **kwargs
        )

    def encode_batch(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """编码文本批次"""
        return self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        """解码token批次"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


class M2M100Wrapper(PretrainedModelWrapper):
    """M2M-100模型包装器"""

    def __init__(
        self,
        model_name: str = 'facebook/m2m100_418M',
        src_lang: str = 'en',
        tgt_lang: str = 'de',
        adapter_dim: int = 64,
        use_adapter: bool = False
    ):
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_adapter = use_adapter
        self.adapter_dim = adapter_dim

        # 加载预训练模型和分词器
        print(f"加载M2M-100模型: {model_name}")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        # 设置源语言
        self.tokenizer.src_lang = src_lang

        # 如果使用适配器，添加适配器层
        if use_adapter:
            self._add_adapters()

    def _add_adapters(self):
        """在模型中添加适配器层"""
        hidden_size = self.model.config.d_model
        self.adapters = nn.ModuleDict()

        # 为编码器每层添加适配器
        for i in range(len(self.model.model.encoder.layers)):
            self.adapters[f'encoder_{i}'] = Adapter(hidden_size, self.adapter_dim)

        # 为解码器每层添加适配器
        for i in range(len(self.model.model.decoder.layers)):
            self.adapters[f'decoder_{i}'] = Adapter(hidden_size, self.adapter_dim)

    def freeze_layers(self, encoder_layers: int = 3, decoder_layers: int = 2):
        """分层冻结策略"""
        # 冻结词嵌入
        for param in self.model.model.shared.parameters():
            param.requires_grad = False

        # 冻结编码器前n层
        for i, layer in enumerate(self.model.model.encoder.layers):
            if i < encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # 冻结解码器前n层
        for i, layer in enumerate(self.model.model.decoder.layers):
            if i < decoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=return_dict
        )
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        num_beams: int = 4,
        **kwargs
    ) -> torch.Tensor:
        """生成翻译"""
        forced_bos_token_id = self.tokenizer.get_lang_id(self.tgt_lang)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            forced_bos_token_id=forced_bos_token_id,
            **kwargs
        )

    def encode_batch(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """编码文本批次"""
        return self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        """解码token批次"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def get_pretrained_model(
    model_type: str,
    src_lang: str = 'en',
    tgt_lang: str = 'de',
    adapter_dim: int = 64,
    use_adapter: bool = False
) -> PretrainedModelWrapper:
    """
    获取预训练模型的工厂函数

    Args:
        model_type: 模型类型
            - 'mbart': mBART-50模型
            - 'm2m100': M2M-100模型
        src_lang: 源语言
        tgt_lang: 目标语言
        adapter_dim: 适配器维度
        use_adapter: 是否使用适配器

    Returns:
        预训练模型包装器
    """
    if model_type == 'mbart':
        # 转换语言代码格式
        src_lang_mbart = 'en_XX' if src_lang == 'en' else f'{src_lang}_XX'
        tgt_lang_mbart = 'de_DE' if tgt_lang == 'de' else f'{tgt_lang}_XX'
        return MBartWrapper(
            src_lang=src_lang_mbart,
            tgt_lang=tgt_lang_mbart,
            adapter_dim=adapter_dim,
            use_adapter=use_adapter
        )
    elif model_type == 'm2m100':
        return M2M100Wrapper(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            adapter_dim=adapter_dim,
            use_adapter=use_adapter
        )
    else:
        raise ValueError(f"未知的预训练模型类型: {model_type}")
