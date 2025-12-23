# -*- coding: utf-8 -*-
"""
数据预处理模块
包含数据加载、清洗、词汇表构建等功能
"""

import re
import os
import hashlib
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy

from config.base_config import DATA_CONFIG, SPECIAL_TOKENS


class Vocabulary:
    """词汇表类，用于管理文本到索引的映射"""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()

        # 添加特殊token
        self._add_special_tokens()

    def _add_special_tokens(self):
        """添加特殊token到词汇表"""
        special_tokens = [
            SPECIAL_TOKENS['pad_token'],
            SPECIAL_TOKENS['sos_token'],
            SPECIAL_TOKENS['eos_token'],
            SPECIAL_TOKENS['unk_token'],
        ]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build_vocab(self, sentences: List[List[str]]):
        """从分词后的句子列表构建词汇表"""
        # 统计词频
        for sentence in sentences:
            self.word_freq.update(sentence)

        # 过滤低频词并添加到词汇表
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, tokens: List[str]) -> List[int]:
        """将token列表转换为索引列表"""
        unk_idx = self.word2idx[SPECIAL_TOKENS['unk_token']]
        return [self.word2idx.get(token, unk_idx) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """将索引列表转换为token列表"""
        return [self.idx2word.get(idx, SPECIAL_TOKENS['unk_token']) for idx in indices]

    def __len__(self):
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[SPECIAL_TOKENS['pad_token']]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[SPECIAL_TOKENS['sos_token']]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[SPECIAL_TOKENS['eos_token']]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[SPECIAL_TOKENS['unk_token']]


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self,
                 max_seq_len: int = DATA_CONFIG['max_seq_len'],
                 min_freq: int = DATA_CONFIG['min_freq'],
                 max_token_len: int = DATA_CONFIG['max_token_len']):
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq
        self.max_token_len = max_token_len

        # 加载spaCy分词器
        try:
            self.en_tokenizer = spacy.load('en_core_web_sm')
            self.de_tokenizer = spacy.load('de_core_news_sm')
        except OSError:
            print("正在下载spaCy模型...")
            os.system('python -m spacy download en_core_web_sm')
            os.system('python -m spacy download de_core_news_sm')
            self.en_tokenizer = spacy.load('en_core_web_sm')
            self.de_tokenizer = spacy.load('de_core_news_sm')

        self.src_vocab: Optional[Vocabulary] = None
        self.tgt_vocab: Optional[Vocabulary] = None

    def clean_text(self, text: str, is_english: bool = True) -> str:
        """
        清洗文本
        - 英文转小写，德语保留原始大小写
        - 仅保留字母、数字、空格、标点符号
        """
        # 去除特殊字符，仅保留字母、数字、空格、标点
        text = re.sub(r"[^a-zA-ZäöüÄÖÜß0-9\s.,!?\"']", '', text)

        # 英文转小写
        if is_english:
            text = text.lower()

        return text.strip()

    def tokenize(self, text: str, is_english: bool = True) -> List[str]:
        """使用spaCy进行分词"""
        tokenizer = self.en_tokenizer if is_english else self.de_tokenizer
        doc = tokenizer(text)
        return [token.text for token in doc]

    def load_multi30k(self) -> Tuple[List, List, List]:
        """加载Multi30k数据集"""
        print("正在加载Multi30k数据集...")

        # 使用Hugging Face datasets加载
        dataset = load_dataset('bentrevett/multi30k')

        train_data = dataset['train']
        val_data = dataset['validation']
        test_data = dataset['test']

        return train_data, val_data, test_data

    def process_dataset(self, data, build_vocab: bool = False) -> List[Dict]:
        """
        处理数据集
        - 清洗文本
        - 分词
        - 过滤异常样本
        - 可选：构建词汇表
        """
        processed_data = []
        src_sentences = []
        tgt_sentences = []

        for item in data:
            en_text = item['en']
            de_text = item['de']

            # 跳过空值
            if not en_text or not de_text:
                continue

            # 清洗文本
            en_clean = self.clean_text(en_text, is_english=True)
            de_clean = self.clean_text(de_text, is_english=False)

            # 分词
            en_tokens = self.tokenize(en_clean, is_english=True)
            de_tokens = self.tokenize(de_clean, is_english=False)

            # 过滤长度超过阈值的样本
            if len(en_tokens) > self.max_token_len or len(de_tokens) > self.max_token_len:
                continue

            processed_data.append({
                'src_tokens': en_tokens,
                'tgt_tokens': de_tokens,
                'src_text': en_clean,
                'tgt_text': de_clean,
            })

            if build_vocab:
                src_sentences.append(en_tokens)
                tgt_sentences.append(de_tokens)

        # 构建词汇表
        if build_vocab:
            self.src_vocab = Vocabulary(min_freq=self.min_freq)
            self.tgt_vocab = Vocabulary(min_freq=self.min_freq)
            self.src_vocab.build_vocab(src_sentences)
            self.tgt_vocab.build_vocab(tgt_sentences)
            print(f"源语言词汇表大小: {len(self.src_vocab)}")
            print(f"目标语言词汇表大小: {len(self.tgt_vocab)}")

        return processed_data

    def encode_data(self, processed_data: List[Dict]) -> List[Dict]:
        """将分词后的数据编码为索引"""
        encoded_data = []

        for item in processed_data:
            src_indices = self.src_vocab.encode(item['src_tokens'])
            tgt_indices = self.tgt_vocab.encode(item['tgt_tokens'])

            # 添加<sos>和<eos>
            src_indices = [self.src_vocab.sos_idx] + src_indices + [self.src_vocab.eos_idx]
            tgt_indices = [self.tgt_vocab.sos_idx] + tgt_indices + [self.tgt_vocab.eos_idx]

            # 截断或填充
            src_indices = src_indices[:self.max_seq_len]
            tgt_indices = tgt_indices[:self.max_seq_len]

            encoded_data.append({
                'src': src_indices,
                'tgt': tgt_indices,
                'src_text': item['src_text'],
                'tgt_text': item['tgt_text'],
                'src_len': len(src_indices),
                'tgt_len': len(tgt_indices),
            })

        return encoded_data

    def get_vocab_md5(self) -> Dict[str, str]:
        """获取词汇表的MD5校验值"""
        src_md5 = hashlib.md5(str(sorted(self.src_vocab.word2idx.items())).encode()).hexdigest()
        tgt_md5 = hashlib.md5(str(sorted(self.tgt_vocab.word2idx.items())).encode()).hexdigest()
        return {'src_vocab_md5': src_md5, 'tgt_vocab_md5': tgt_md5}

    def save_vocab(self, save_dir: str):
        """保存词汇表"""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'src_vocab.pkl'), 'wb') as f:
            pickle.dump(self.src_vocab, f)
        with open(os.path.join(save_dir, 'tgt_vocab.pkl'), 'wb') as f:
            pickle.dump(self.tgt_vocab, f)

        # 保存MD5校验值
        md5_info = self.get_vocab_md5()
        with open(os.path.join(save_dir, 'vocab_md5.txt'), 'w') as f:
            f.write(f"src_vocab_md5: {md5_info['src_vocab_md5']}\n")
            f.write(f"tgt_vocab_md5: {md5_info['tgt_vocab_md5']}\n")

        print(f"词汇表已保存至 {save_dir}")

    def load_vocab(self, save_dir: str):
        """加载词汇表"""
        with open(os.path.join(save_dir, 'src_vocab.pkl'), 'rb') as f:
            self.src_vocab = pickle.load(f)
        with open(os.path.join(save_dir, 'tgt_vocab.pkl'), 'rb') as f:
            self.tgt_vocab = pickle.load(f)
        print(f"词汇表已从 {save_dir} 加载")


def collate_fn(batch: List[Dict], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    批次整理函数
    将变长序列填充为相同长度
    """
    src_batch = [torch.tensor(item['src']) for item in batch]
    tgt_batch = [torch.tensor(item['tgt']) for item in batch]

    # 填充序列
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    # 创建mask
    src_mask = (src_padded != pad_idx)
    tgt_mask = (tgt_padded != pad_idx)

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
    }


if __name__ == '__main__':
    # 测试预处理流程
    preprocessor = DataPreprocessor()
    train_data, val_data, test_data = preprocessor.load_multi30k()

    # 处理训练集并构建词汇表
    train_processed = preprocessor.process_dataset(train_data, build_vocab=True)
    val_processed = preprocessor.process_dataset(val_data, build_vocab=False)
    test_processed = preprocessor.process_dataset(test_data, build_vocab=False)

    # 编码数据
    train_encoded = preprocessor.encode_data(train_processed)
    val_encoded = preprocessor.encode_data(val_processed)
    test_encoded = preprocessor.encode_data(test_processed)

    print(f"训练集样本数: {len(train_encoded)}")
    print(f"验证集样本数: {len(val_encoded)}")
    print(f"测试集样本数: {len(test_encoded)}")

    # 保存词汇表
    preprocessor.save_vocab('./vocab')
