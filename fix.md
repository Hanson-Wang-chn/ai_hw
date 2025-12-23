你刚刚按照 doc.md 中的要求完成了代码编写。在运行第3个实验时，出现了下面的错误，请帮我完善代码，确保可以正常运行。如果该问题重复出现在后续实验中，也请改正所有的地方，确保所有的代码都能正常运行（不要修改该实验之前的实验的代码）。**注意当前是pytorch 2.7.1版本。**

```
(ai_hw) whs@whs-OMEN:~/ai_hw/code$ python run.py --task exp3_feedforward
============================================================
运行实验: exp3_feedforward
输出目录: /home/whs/ai_hw/code/exp3_feedforward/results
============================================================
[nltk_data] Downloading package wordnet to /home/whs/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package omw-1.4 to /home/whs/nltk_data...
[nltk_data]   Package omw-1.4 is already up-to-date!
准备数据...
加载缓存数据...
词汇表已从 ./data_cache/vocab 加载
训练集: 28920 样本
验证集: 1011 样本
测试集: 998 样本

==================================================
运行实验: ffn_relu_4x
==================================================
模型参数量: 33,240,849
2025-12-23 16:19:18 - INFO - 开始训练，共 50 轮
2025-12-23 16:19:18 - INFO - 模型参数量: 33,240,849
Epoch 1 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.31it/s, loss=1.8454]
Epoch 1 [Val]: 100%|███████████████████████████| 32/32 [00:01<00:00, 16.17it/s, loss=0.8064]
2025-12-23 16:19:49 - INFO - Epoch 1: train_loss=2.8214, val_loss=0.5686, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 1: train_loss=2.8214, val_loss=0.5686, val_bleu_4=0.0044, lr=1.00e-04
2025-12-23 16:19:49 - INFO - 新的最佳模型！BLEU-4: 0.0044
INFO:train:新的最佳模型！BLEU-4: 0.0044
Epoch 2 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.48it/s, loss=1.6038]
Epoch 2 [Val]: 100%|███████████████████████████| 32/32 [00:01<00:00, 16.15it/s, loss=0.4533]
2025-12-23 16:20:19 - INFO - Epoch 2: train_loss=1.7009, val_loss=0.3035, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 2: train_loss=1.7009, val_loss=0.3035, val_bleu_4=0.0044, lr=1.00e-04
Epoch 3 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.24it/s, loss=1.4248]
Epoch 3 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.73it/s, loss=0.3266]
2025-12-23 16:20:49 - INFO - Epoch 3: train_loss=1.5193, val_loss=0.2155, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 3: train_loss=1.5193, val_loss=0.2155, val_bleu_4=0.0044, lr=1.00e-04
Epoch 4 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.31it/s, loss=1.4065]
Epoch 4 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.55it/s, loss=0.2726]
2025-12-23 16:21:18 - INFO - Epoch 4: train_loss=1.4374, val_loss=0.1735, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 4: train_loss=1.4374, val_loss=0.1735, val_bleu_4=0.0044, lr=1.00e-04
Epoch 5 [Train]: 100%|███████████████████████| 904/904 [00:25<00:00, 35.83it/s, loss=1.4213]
Epoch 5 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.47it/s, loss=0.2324]
2025-12-23 16:21:49 - INFO - Epoch 5: train_loss=1.3916, val_loss=0.1503, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 5: train_loss=1.3916, val_loss=0.1503, val_bleu_4=0.0044, lr=1.00e-04
Epoch 6 [Train]: 100%|███████████████████████| 904/904 [00:25<00:00, 36.16it/s, loss=1.3416]
Epoch 6 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.45it/s, loss=0.2199]
2025-12-23 16:22:19 - INFO - Epoch 6: train_loss=1.3614, val_loss=0.1424, val_bleu_4=0.0044, lr=5.00e-05
INFO:train:Epoch 6: train_loss=1.3614, val_loss=0.1424, val_bleu_4=0.0044, lr=5.00e-05
2025-12-23 16:22:19 - INFO - 早停触发，已有 5 轮无提升
INFO:train:早停触发，已有 5 轮无提升
2025-12-23 16:22:19 - INFO - 训练完成，耗时: 3.02 分钟
INFO:train:训练完成，耗时: 3.02 分钟
2025-12-23 16:22:19 - INFO - 在测试集上评估最佳模型...
INFO:train:在测试集上评估最佳模型...
2025-12-23 16:22:19 - INFO - 加载检查点: /home/whs/ai_hw/code/exp3_feedforward/results/ffn_relu_4x/checkpoints/best_model.pth
INFO:train:加载检查点: /home/whs/ai_hw/code/exp3_feedforward/results/ffn_relu_4x/checkpoints/best_model.pth
Test: 100%|████████████████████████████████████| 32/32 [00:02<00:00, 15.81it/s, loss=0.5470]
2025-12-23 16:22:24 - INFO - 测试集结果: {'bleu_1': np.float64(0.02468094082902647), 'bleu_2': np.float64(0.008018668483230665), 'bleu_4': np.float64(0.004774925052524541), 'rouge_l': np.float64(0.06117487462422324), 'bert_score': 0.4818650186061859, 'loss': 0.5732011385262012}
INFO:train:测试集结果: {'bleu_1': np.float64(0.02468094082902647), 'bleu_2': np.float64(0.008018668483230665), 'bleu_4': np.float64(0.004774925052524541), 'rouge_l': np.float64(0.06117487462422324), 'bert_score': 0.4818650186061859, 'loss': 0.5732011385262012}
图片已保存至: /home/whs/ai_hw/code/exp3_feedforward/results/ffn_relu_4x/img/training_curves.png
图片已保存至: /home/whs/ai_hw/code/exp3_feedforward/results/ffn_relu_4x/img/gradient_norms.png
图片已保存至: /home/whs/ai_hw/code/exp3_feedforward/results/ffn_relu_4x/img/feature_tsne.png
Traceback (most recent call last):
  File "/home/whs/ai_hw/code/run.py", line 142, in <module>
    main()
  File "/home/whs/ai_hw/code/run.py", line 102, in main
    run_experiment()
  File "/home/whs/ai_hw/code/exp3_feedforward/run_exp3.py", line 252, in run_experiment
    results = run_single_experiment(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/exp3_feedforward/run_exp3.py", line 113, in run_single_experiment
    json.dump(results, f, indent=2, ensure_ascii=False)
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/json/__init__.py", line 179, in dump
    for chunk in iterable:
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/json/encoder.py", line 432, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/json/encoder.py", line 439, in _iterencode
    o = _default(o)
        ^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type float32 is not JSON serializable
(ai_hw) whs@whs-OMEN:~/ai_hw/code$ 
```