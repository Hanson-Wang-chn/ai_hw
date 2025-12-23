# -*- coding: utf-8 -*-
"""
数据增强模块
实现回译、混合语言对、句子扰动等数据增强方法
"""

import random
from typing import List, Dict, Optional, Tuple
import nltk
from nltk.corpus import wordnet

# 确保下载必要的NLTK数据
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


class BackTranslation:
    """
    回译数据增强
    使用翻译模型将目标语言翻译回源语言，生成伪平行句对
    """

    def __init__(self, model_name: str = 'facebook/mbart-large-50-many-to-many-mmt'):
        """
        Args:
            model_name: 用于回译的预训练模型名称
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """延迟加载模型"""
        if self.model is None:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

            print(f"加载回译模型: {self.model_name}")
            self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
            self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)

            import torch
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        翻译文本

        Args:
            texts: 输入文本列表
            src_lang: 源语言代码 (如 'de_DE', 'en_XX')
            tgt_lang: 目标语言代码

        Returns:
            翻译后的文本列表
        """
        import torch

        self.load_model()

        self.tokenizer.src_lang = src_lang
        translations = []

        # 批量处理
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                    max_length=50
                )

            batch_translations = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations.extend(batch_translations)

        return translations

    def augment(self, data: List[Dict], ratio: float = 1.0) -> List[Dict]:
        """
        通过回译进行数据增强

        Args:
            data: 原始数据列表
            ratio: 增强比例

        Returns:
            增强后的数据列表
        """
        num_samples = int(len(data) * ratio)
        sampled_data = random.sample(data, min(num_samples, len(data)))

        # 提取德语目标句
        de_texts = [item['tgt_text'] for item in sampled_data]

        # 德语 -> 英语回译
        en_translations = self.translate(de_texts, src_lang='de_DE', tgt_lang='en_XX')

        # 构建伪平行句对
        augmented = []
        for i, item in enumerate(sampled_data):
            augmented.append({
                'src_text': en_translations[i].lower(),  # 英文转小写
                'tgt_text': item['tgt_text'],
                'is_augmented': True,
                'aug_type': 'back_translation'
            })

        return augmented


class SentenceDisturbance:
    """
    句子扰动数据增强
    包括同义替换、随机删除、句子分割
    """

    def __init__(self, delete_ratio: float = 0.1):
        """
        Args:
            delete_ratio: 随机删除比例
        """
        self.delete_ratio = delete_ratio

    def synonym_replacement(self, tokens: List[str], n: int = 2) -> List[str]:
        """
        同义替换

        Args:
            tokens: 输入token列表
            n: 替换数量

        Returns:
            替换后的token列表
        """
        new_tokens = tokens.copy()
        random_word_list = list(set([word for word in tokens if word.isalpha()]))
        random.shuffle(random_word_list)

        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) > 0:
                synonym = random.choice(list(synonyms))
                new_tokens = [synonym if word == random_word else word for word in new_tokens]
                num_replaced += 1
            if num_replaced >= n:
                break

        return new_tokens

    def _get_synonyms(self, word: str) -> set:
        """获取同义词"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
        return synonyms

    def random_deletion(self, tokens: List[str]) -> List[str]:
        """
        随机删除非核心词汇

        Args:
            tokens: 输入token列表

        Returns:
            删除后的token列表
        """
        if len(tokens) <= 3:
            return tokens

        # 保留的词（避免删除过多）
        new_tokens = []
        for token in tokens:
            if random.random() > self.delete_ratio:
                new_tokens.append(token)

        # 确保至少保留一半的词
        if len(new_tokens) < len(tokens) // 2:
            return tokens[:len(tokens) // 2]

        return new_tokens if new_tokens else tokens

    def sentence_split(self, tokens: List[str]) -> List[List[str]]:
        """
        简单句分割（基于标点符号）

        Args:
            tokens: 输入token列表

        Returns:
            分割后的句子列表
        """
        splits = []
        current = []

        for token in tokens:
            current.append(token)
            if token in [',', '.', ';', '!', '?']:
                if len(current) > 2:  # 确保分割后的句子有意义
                    splits.append(current[:-1])  # 不包含标点
                current = []

        if current and len(current) > 2:
            splits.append(current)

        return splits if splits else [tokens]

    def augment(self, data: List[Dict], ratio: float = 3.0) -> List[Dict]:
        """
        通过句子扰动进行数据增强

        Args:
            data: 原始数据列表
            ratio: 增强比例（每个样本生成ratio个变体）

        Returns:
            增强后的数据列表
        """
        augmented = []

        for item in data:
            tokens = item['src_text'].split()

            # 同义替换
            syn_tokens = self.synonym_replacement(tokens)
            augmented.append({
                'src_text': ' '.join(syn_tokens),
                'tgt_text': item['tgt_text'],
                'is_augmented': True,
                'aug_type': 'synonym_replacement'
            })

            # 随机删除
            del_tokens = self.random_deletion(tokens)
            augmented.append({
                'src_text': ' '.join(del_tokens),
                'tgt_text': item['tgt_text'],
                'is_augmented': True,
                'aug_type': 'random_deletion'
            })

            # 句子分割（仅对较长句子）
            if len(tokens) > 8:
                splits = self.sentence_split(tokens)
                if len(splits) > 1:
                    # 取第一个分割部分
                    augmented.append({
                        'src_text': ' '.join(splits[0]),
                        'tgt_text': item['tgt_text'],  # 简化处理，保留原目标
                        'is_augmented': True,
                        'aug_type': 'sentence_split'
                    })
                else:
                    # 如果无法分割，再做一次同义替换
                    syn_tokens2 = self.synonym_replacement(tokens, n=3)
                    augmented.append({
                        'src_text': ' '.join(syn_tokens2),
                        'tgt_text': item['tgt_text'],
                        'is_augmented': True,
                        'aug_type': 'synonym_replacement_2'
                    })
            else:
                # 短句子做双重同义替换
                syn_tokens2 = self.synonym_replacement(tokens, n=3)
                augmented.append({
                    'src_text': ' '.join(syn_tokens2),
                    'tgt_text': item['tgt_text'],
                    'is_augmented': True,
                    'aug_type': 'synonym_replacement_2'
                })

        return augmented


class MixedLanguageAugmentation:
    """
    混合语言对数据增强
    添加法语-德语平行语料作为辅助
    """

    def __init__(self):
        self.fr_de_data = None

    def load_auxiliary_data(self) -> List[Dict]:
        """
        加载辅助语料（模拟法语-德语数据）
        实际使用时应加载真实的法德平行语料
        """
        # 这里使用WMT等数据集中的法德数据
        # 为简化实现，从现有数据生成模拟数据
        print("注意：混合语言对增强需要额外的法德平行语料")
        return []

    def augment(self, data: List[Dict], mix_ratio: float = 0.2) -> List[Dict]:
        """
        混合语言对增强

        Args:
            data: 原始英德数据
            mix_ratio: 混合比例

        Returns:
            混合后的数据
        """
        # 加载辅助数据
        if self.fr_de_data is None:
            self.fr_de_data = self.load_auxiliary_data()

        # 计算需要添加的辅助数据量
        num_aux = int(len(data) * mix_ratio / (1 - mix_ratio))

        if len(self.fr_de_data) > 0:
            sampled_aux = random.sample(self.fr_de_data, min(num_aux, len(self.fr_de_data)))
            return data + sampled_aux
        else:
            print("警告：无可用辅助语料，返回原始数据")
            return data


def filter_augmented_data(
    augmented_data: List[Dict],
    min_bleu: float = 0.6,
    reference_model=None
) -> List[Dict]:
    """
    过滤增强数据质量

    Args:
        augmented_data: 增强后的数据
        min_bleu: 最小BLEU阈值
        reference_model: 参考模型（用于评估质量）

    Returns:
        过滤后的数据
    """
    # 简化实现：基于启发式规则过滤
    filtered = []
    for item in augmented_data:
        src_text = item['src_text']

        # 过滤过短或过长的句子
        tokens = src_text.split()
        if len(tokens) < 3 or len(tokens) > 25:
            continue

        # 过滤重复词过多的句子
        if len(set(tokens)) < len(tokens) * 0.5:
            continue

        filtered.append(item)

    print(f"增强数据过滤: {len(augmented_data)} -> {len(filtered)}")
    return filtered


class DataAugmentor:
    """数据增强器统一接口"""

    def __init__(self, method: str = 'none'):
        """
        Args:
            method: 增强方法
                - 'none': 无增强
                - 'back_translation': 回译
                - 'sentence_disturb': 句子扰动
                - 'mix_language': 混合语言对
        """
        self.method = method
        self.augmentor = None

        if method == 'back_translation':
            self.augmentor = BackTranslation()
        elif method == 'sentence_disturb':
            self.augmentor = SentenceDisturbance()
        elif method == 'mix_language':
            self.augmentor = MixedLanguageAugmentation()

    def augment(self, data: List[Dict], **kwargs) -> List[Dict]:
        """
        执行数据增强

        Args:
            data: 原始数据
            **kwargs: 增强参数

        Returns:
            增强后的数据（包含原始数据）
        """
        if self.method == 'none' or self.augmentor is None:
            return data

        augmented = self.augmentor.augment(data, **kwargs)

        # 过滤质量不佳的增强数据
        augmented = filter_augmented_data(augmented)

        # 合并原始数据和增强数据
        return data + augmented
