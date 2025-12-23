# Seq2Seq模型设计与实现——人工智能导论实验

本项目实现了基于Transformer和RNN的Seq2Seq模型，用于英德翻译任务。项目包含7个对比实验，系统性地探究了位置编码、注意力机制、FeedForward网络、归一化策略、解码策略、预训练微调和数据增强等因素对翻译性能的影响。

## 项目结构

```
code/
├── config/                          # 配置模块
│   ├── __init__.py
│   └── base_config.py               # 基准超参数配置
├── data/                            # 数据处理模块
│   ├── __init__.py
│   ├── preprocess.py                # 数据预处理（含词汇表构建）
│   ├── dataset.py                   # 自定义Dataset类
│   └── data_augmentation.py         # 数据增强方法
├── model/                           # 模型模块
│   ├── __init__.py
│   ├── base_transformer.py          # 基准Transformer实现
│   ├── pe_module.py                 # 位置编码模块
│   ├── attention_module.py          # 注意力机制模块
│   ├── rnn_seq2seq.py               # RNN基准模型
│   └── pretrain_model.py            # 预训练模型封装
├── train/                           # 训练模块
│   ├── __init__.py
│   ├── trainer.py                   # 通用训练器
│   ├── finetune_trainer.py          # 微调训练器
│   └── utils.py                     # 工具函数
├── infer/                           # 推理模块
│   ├── __init__.py
│   ├── decoder.py                   # 解码策略实现
│   └── infer.py                     # 推理脚本
├── visualization/                   # 可视化模块
│   ├── __init__.py
│   ├── plot_utils.py                # 绘图工具
│   └── analyze.py                   # 结果分析
├── exp1_position_encoding/          # 实验一：位置编码对比
│   ├── configs/                     # 配置文件
│   │   ├── sinusoidal.yaml
│   │   ├── learnable.yaml
│   │   └── relative.yaml
│   ├── run_exp1.py                  # 运行脚本
│   └── img/                         # 可视化结果
├── exp2_attention/                  # 实验二：注意力机制对比
│   ├── configs/
│   │   ├── scaled_dot.yaml
│   │   ├── multi_head.yaml
│   │   ├── linear.yaml
│   │   └── bidirectional.yaml
│   ├── run_exp2.py
│   └── img/
├── exp3_feedforward/                # 实验三：FFN结构对比
│   ├── configs/
│   │   ├── relu_4x.yaml
│   │   ├── gelu_4x.yaml
│   │   └── relu_2x.yaml
│   ├── run_exp3.py
│   └── img/
├── exp4_normalization/              # 实验四：归一化策略对比
│   ├── configs/
│   │   ├── post_ln.yaml
│   │   ├── pre_ln.yaml
│   │   └── mixed.yaml
│   ├── run_exp4.py
│   └── img/
├── exp5_decoding/                   # 实验五：解码策略优化
│   ├── configs/
│   │   ├── greedy.yaml
│   │   ├── beam.yaml
│   │   └── sampling.yaml
│   ├── run_exp5.py
│   └── img/
├── exp6_pretrain/                   # 实验六：预训练与微调
│   ├── configs/
│   │   ├── no_pretrain.yaml
│   │   ├── mbart_full.yaml
│   │   ├── mbart_freeze.yaml
│   │   └── mbart_adapter.yaml
│   ├── run_exp6.py
│   └── img/
├── exp7_data_augmentation/          # 实验七：数据增强
│   ├── configs/
│   │   ├── no_aug.yaml
│   │   ├── back_translation.yaml
│   │   └── sentence_disturb.yaml
│   ├── run_exp7.py
│   └── img/
├── run.py                           # 统一运行入口
├── requirements.txt                 # 依赖清单
└── README.md                        # 本文档
```

## 环境要求

- **操作系统**: Ubuntu 22.04
- **Python版本**: 3.11
- **PyTorch版本**: 2.7.1
- **GPU**: NVIDIA GPU（显存 >= 14GB）

## 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 下载spaCy语言模型
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

# 下载NLTK数据
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 运行方法

### 统一入口

使用 `run.py` 运行任意实验：

```bash
# 运行实验一（位置编码对比），执行所有配置
python run.py --task exp1_position_encoding

# 运行实验一，仅执行指定配置
python run.py --task exp1_position_encoding --config sinusoidal.yaml

# 运行实验二（注意力机制对比）
python run.py --task exp2_attention

# 运行实验三（FFN结构对比）
python run.py --task exp3_feedforward

# 运行实验四（归一化策略对比）
python run.py --task exp4_normalization

# 运行实验五（解码策略优化）
python run.py --task exp5_decoding

# 运行实验六（预训练与微调）
python run.py --task exp6_pretrain

# 运行实验七（数据增强）
python run.py --task exp7_data_augmentation

# 指定输出目录
python run.py --task exp1_position_encoding --output_dir ./my_results
```

### 直接运行单个实验

```bash
cd code
python exp1_position_encoding/run_exp1.py
python exp2_attention/run_exp2.py
# ...以此类推
```

## 实验说明

### 实验一：位置编码对比（exp1_position_encoding）

**实验目的**：探究不同位置编码方式对Transformer翻译性能的影响。

**实验设置**：
- PE1：标准正弦位置编码（Vaswani原版，基准组）
- PE2：可学习位置编码（嵌入层维度与词嵌入一致）
- PE3：相对位置编码（Shaw et al., 2018）

**评估指标**：BLEU-1/2/4、ROUGE-L、BERTScore、收敛速度、长句翻译准确率

**可视化输出**：
- `img/training_curves.png`：训练/验证损失曲线
- `img/pe_similarity_*.png`：位置编码相似度热力图

**结果保存**：
- `results.json`：包含所有评估指标
- `checkpoints/best_model.pth`：最优模型权重

### 实验二：注意力机制对比（exp2_attention）

**实验目的**：分析不同注意力机制对模型捕捉语义关联的影响。

**实验设置**：
- ATT1：标准Scaled Dot-Product Attention
- ATT2：Multi-Head Attention（8头）
- ATT3：Linear Attention（轻量化）
- ATT4：双向注意力

**额外指标**：注意力权重分布熵、训练时间

**可视化输出**：
- `img/attention_sample_*.png`：注意力热力图

### 实验三：FeedForward网络对比（exp3_feedforward）

**实验目的**：探究FFN的激活函数、隐藏层维度对特征提取能力的影响。

**实验设置**：
- FFN1：ReLU + 4倍隐藏层（基准组）
- FFN2：GELU + 4倍隐藏层
- FFN3：ReLU + 2倍隐藏层（轻量化）

**额外指标**：特征提取能力评分、梯度消失/爆炸发生率

**可视化输出**：
- `img/gradient_norms.png`：梯度范数演化曲线
- `img/feature_tsne.png`：特征聚类散点图

### 实验四：归一化策略对比（exp4_normalization）

**实验目的**：验证Pre-LN与Post-LN对训练稳定性和收敛速度的影响。

**实验设置**：
- NORM1：后归一化（Post-LN，基准组）
- NORM2：预归一化（Pre-LN）
- NORM3：混合归一化（Encoder用Pre-LN，Decoder用Post-LN）

**额外指标**：训练稳定性评分（损失方差）、收敛速度

**可视化输出**：
- `img/training_curves.png`：带置信区间的损失曲线

### 实验五：解码策略优化（exp5_decoding）

**实验目的**：探究不同解码策略对生成文本质量的影响。

**实验设置**：
- DEC1：贪心搜索（基准组）
- DEC2：束搜索（束宽=2/4/6/8）
- DEC3：温度采样（温度=0.5/1.0/1.5）

**额外指标**：生成多样性评分、解码速度

**可视化输出**：
- `img/beam_width_performance.png`：束宽-性能曲线
- `img/diversity_vs_quality.png`：多样性vs质量散点图
- `img/prob_dist_step_*.png`：解码概率分布

### 实验六：预训练与微调（exp6_pretrain）

**实验目的**：探究预训练+微调策略在小样本翻译任务中的有效性。

**实验设置**：
- PT1：无预训练（基准组）
- PT2：mBART + 全参数微调
- PT3：mBART + 分层冻结
- PT4：mBART + 适配器微调

**额外指标**：收敛速度、灾难性遗忘程度、训练效率

**可视化输出**：
- `img/training_curves.png`：预训练-微调损失演化曲线

### 实验七：数据增强（exp7_data_augmentation）

**实验目的**：探究不同数据增强方法对性能提升效果。

**实验设置**：
- DAug1：无数据增强（基准组）
- DAug2：回译增强（扩增比例1:1）
- DAug3：句子扰动增强（扩增比例1:3）

**额外指标**：模型鲁棒性评分、增强数据有效性评分

**可视化输出**：
- `img/data_distribution_*.png`：数据分布对比散点图

## 数据预处理

数据预处理流程（参见 `data/preprocess.py`）：

1. **数据加载**：使用Hugging Face Datasets加载Multi30k数据集
2. **文本清洗**：
   - 英文转小写，德语保留原始大小写
   - 仅保留字母、数字、空格、标点符号
3. **分词**：使用spaCy分词器（en_core_web_sm、de_core_news_sm）
4. **词汇表构建**：最小词频阈值=2，添加特殊token（\<pad\>、\<sos\>、\<eos\>、\<unk\>）
5. **序列处理**：统一序列长度为20，使用padding_mask处理填充

**词汇表MD5校验**：运行预处理后，校验值保存在 `data_cache/vocab/vocab_md5.txt`

## 模型配置

所有实验的固定超参数（参见 `config/base_config.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 512 | 模型维度 |
| nhead | 8 | 注意力头数 |
| num_encoder_layers | 3 | 编码器层数 |
| num_decoder_layers | 3 | 解码器层数 |
| dim_feedforward | 2048 | FFN隐藏层维度 |
| dropout | 0.1 | Dropout概率 |
| lr | 1e-4 | 学习率 |
| batch_size | 32 | 批次大小 |
| epochs | 50 | 训练轮次 |
| patience | 5 | 早停patience |

## 评估指标

- **BLEU-1/2/4**：基于n-gram的翻译质量评估
- **ROUGE-L**：基于最长公共子序列的评估
- **BERTScore**：基于BERT的语义相似度评估
- **注意力熵**：衡量注意力分布的集中程度
- **梯度范数**：衡量训练稳定性

## 日志与可视化

- **TensorBoard日志**：保存在 `<exp>/results/<config>/tensorboard/`
- **文本日志**：保存在 `<exp>/results/<config>/logs/`
- **可视化图片**：保存在 `<exp>/results/<config>/img/`

查看TensorBoard：
```bash
tensorboard --logdir=exp1_position_encoding/results/
```

## 显存优化

为确保VRAM占用不超过14GB，采取以下措施：

1. 使用混合精度训练（AMP）
2. 梯度累积（可选）
3. 适当的batch_size设置
4. 及时清理中间变量

## 注意事项

1. 首次运行会自动下载Multi30k数据集和spaCy模型
2. 预训练实验（exp6）需要较大显存，建议降低batch_size
3. 回译数据增强（exp7）需要较长时间
4. 所有实验支持断点续训，只需重新运行即可

## 参考文献

1. Vaswani et al., "Attention Is All You Need", 2017
2. Shaw et al., "Self-Attention with Relative Position Representations", 2018
3. Hendrycks et al., "Gaussian Error Linear Units (GELUs)", 2016
4. Choromanski et al., "Rethinking Attention with Performers", 2020
5. Xiong et al., "On Layer Normalization in the Transformer Architecture", 2020
