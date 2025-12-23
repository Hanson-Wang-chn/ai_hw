你刚刚按照 doc.md 中的要求完成了代码编写。在运行第一个实验时，出现了下面的错误，请帮我完善代码，确保可以正常运行。如果该问题重复出现在后续实验中，也请改正所有的地方，确保所有的代码都能正常运行。**注意当前是pytorch 2.7.1版本。**

```
(ai_hw) whs@whs-OMEN:~/ai_hw/code$ python run.py --task exp1_position_encoding
============================================================
运行实验: exp1_position_encoding
输出目录: /home/whs/ai_hw/code/exp1_position_encoding/results
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
运行实验: position_encoding_relative
==================================================
模型参数量: 43,479,825
/home/whs/ai_hw/code/train/trainer.py:83: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = GradScaler() if self.use_amp else None
2025-12-23 15:05:24 - INFO - 开始训练，共 50 轮
2025-12-23 15:05:24 - INFO - 模型参数量: 43,479,825
Epoch 1 [Train]:   0%|                                              | 0/904 [00:00<?, ?it/s]/home/whs/ai_hw/code/train/trainer.py:133: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
Epoch 1 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.86it/s, loss=1.5985]
Epoch 1 [Val]: 100%|███████████████████████████| 32/32 [00:01<00:00, 16.51it/s, loss=0.6622]
2025-12-23 15:05:54 - INFO - Epoch 1: train_loss=2.5047, val_loss=0.4426, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 1: train_loss=2.5047, val_loss=0.4426, val_bleu_4=0.0044, lr=1.00e-04
2025-12-23 15:05:54 - INFO - 新的最佳模型！BLEU-4: 0.0044
INFO:train:新的最佳模型！BLEU-4: 0.0044
Epoch 2 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.84it/s, loss=1.4572]
Epoch 2 [Val]: 100%|███████████████████████████| 32/32 [00:01<00:00, 16.21it/s, loss=0.3695]
2025-12-23 15:06:25 - INFO - Epoch 2: train_loss=1.5744, val_loss=0.2408, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 2: train_loss=1.5744, val_loss=0.2408, val_bleu_4=0.0044, lr=1.00e-04
Epoch 3 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.55it/s, loss=1.4427]
Epoch 3 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.92it/s, loss=0.2714]
2025-12-23 15:06:55 - INFO - Epoch 3: train_loss=1.4336, val_loss=0.1709, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 3: train_loss=1.4336, val_loss=0.1709, val_bleu_4=0.0044, lr=1.00e-04
Epoch 4 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.55it/s, loss=1.3599]
Epoch 4 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.97it/s, loss=0.2218]
2025-12-23 15:07:25 - INFO - Epoch 4: train_loss=1.3745, val_loss=0.1424, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 4: train_loss=1.3745, val_loss=0.1424, val_bleu_4=0.0044, lr=1.00e-04
Epoch 5 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.74it/s, loss=1.3859]
Epoch 5 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.86it/s, loss=0.1926]
2025-12-23 15:07:55 - INFO - Epoch 5: train_loss=1.3436, val_loss=0.1279, val_bleu_4=0.0044, lr=1.00e-04
INFO:train:Epoch 5: train_loss=1.3436, val_loss=0.1279, val_bleu_4=0.0044, lr=1.00e-04
Epoch 6 [Train]: 100%|███████████████████████| 904/904 [00:24<00:00, 36.66it/s, loss=1.2498]
Epoch 6 [Val]: 100%|███████████████████████████| 32/32 [00:02<00:00, 15.81it/s, loss=0.1996]
2025-12-23 15:08:25 - INFO - Epoch 6: train_loss=1.3225, val_loss=0.1249, val_bleu_4=0.0044, lr=5.00e-05
INFO:train:Epoch 6: train_loss=1.3225, val_loss=0.1249, val_bleu_4=0.0044, lr=5.00e-05
2025-12-23 15:08:25 - INFO - 早停触发，已有 5 轮无提升
INFO:train:早停触发，已有 5 轮无提升
2025-12-23 15:08:25 - INFO - 训练完成，耗时: 3.01 分钟
INFO:train:训练完成，耗时: 3.01 分钟
2025-12-23 15:08:25 - INFO - 在测试集上评估最佳模型...
INFO:train:在测试集上评估最佳模型...
2025-12-23 15:08:25 - INFO - 加载检查点: /home/whs/ai_hw/code/exp1_position_encoding/results/position_encoding_relative/checkpoints/best_model.pth
INFO:train:加载检查点: /home/whs/ai_hw/code/exp1_position_encoding/results/position_encoding_relative/checkpoints/best_model.pth
Test: 100%|████████████████████████████████████| 32/32 [00:01<00:00, 16.19it/s, loss=0.4404]
2025-12-23 15:08:31 - INFO - 测试集结果: {'bleu_1': np.float64(0.02468094082902647), 'bleu_2': np.float64(0.008018668483230665), 'bleu_4': np.float64(0.004774925052524541), 'rouge_l': np.float64(0.06117487462422324), 'bert_score': 0.4818650186061859, 'loss': 0.44937722105532885}
INFO:train:测试集结果: {'bleu_1': np.float64(0.02468094082902647), 'bleu_2': np.float64(0.008018668483230665), 'bleu_4': np.float64(0.004774925052524541), 'rouge_l': np.float64(0.06117487462422324), 'bert_score': 0.4818650186061859, 'loss': 0.44937722105532885}
Traceback (most recent call last):
  File "/home/whs/ai_hw/code/run.py", line 142, in <module>
    main()
  File "/home/whs/ai_hw/code/run.py", line 88, in main
    run_experiment()
  File "/home/whs/ai_hw/code/exp1_position_encoding/run_exp1.py", line 253, in run_experiment
    results = run_single_experiment(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/exp1_position_encoding/run_exp1.py", line 87, in run_single_experiment
    plot_training_curves(
  File "/home/whs/ai_hw/code/visualization/plot_utils.py", line 196, in plot_training_curves
    fig, axes = plt.subplots(1, 2 if val_metric else 1, figsize=(14 if val_metric else 7, 5))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/pyplot.py", line 1776, in subplots
    fig = figure(**fig_kw)
          ^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/pyplot.py", line 1041, in figure
    manager = new_figure_manager(
              ^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/pyplot.py", line 551, in new_figure_manager
    return _get_backend_mod().new_figure_manager(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backend_bases.py", line 3504, in new_figure_manager
    return cls.new_figure_manager_given_figure(num, fig)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backend_bases.py", line 3509, in new_figure_manager_given_figure
    return cls.FigureCanvas.new_manager(figure, num)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backend_bases.py", line 1785, in new_manager
    return cls.manager_class.create_with_canvas(cls, figure, num)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backends/_backend_tk.py", line 556, in create_with_canvas
    manager = cls(canvas, num, window)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backends/_backend_tk.py", line 509, in __init__
    super().__init__(canvas, num)
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backend_bases.py", line 2630, in __init__
    self.toolbar = self._toolbar2_class(self.canvas)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backends/_backend_tk.py", line 676, in __init__
    self._buttons[text] = button = self._Button(
                                   ^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backends/_backend_tk.py", line 877, in _Button
    NavigationToolbar2Tk._set_image_for_button(self, b)
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/matplotlib/backends/_backend_tk.py", line 813, in _set_image_for_button
    image = ImageTk.PhotoImage(im.resize((size, size)), master=self)
                               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/PIL/Image.py", line 2281, in resize
    im = im.resize(size, resample, box)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/PIL/Image.py", line 2304, in resize
    return self._new(self.im.resize(size, resample, box))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: height and width must be > 0
(ai_hw) whs@whs-OMEN:~/ai_hw/code$ 
```