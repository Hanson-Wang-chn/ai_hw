你刚刚按照 doc.md 中的要求完成了代码编写。在运行第2个实验时，出现了下面的错误，请帮我完善代码，确保可以正常运行。如果该问题重复出现在后续实验中，也请改正所有的地方，确保所有的代码都能正常运行（不要修改该实验之前的实验的代码）。**注意当前是pytorch 2.7.1版本。**

```
==================================================
运行实验: attention_linear
==================================================
模型参数量: 33,240,849
2025-12-23 15:52:00 - INFO - 开始训练，共 50 轮
INFO:train:开始训练，共 50 轮
2025-12-23 15:52:00 - INFO - 模型参数量: 33,240,849
INFO:train:模型参数量: 33,240,849
Epoch 1 [Train]:   0%|                                              | 0/904 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/whs/ai_hw/code/run.py", line 142, in <module>
    main()
  File "/home/whs/ai_hw/code/run.py", line 95, in main
    run_experiment()
  File "/home/whs/ai_hw/code/exp2_attention/run_exp2.py", line 221, in run_experiment
    results = run_single_experiment(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/exp2_attention/run_exp2.py", line 84, in run_single_experiment
    history = trainer.train(train_config['epochs'])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/train/trainer.py", line 285, in train
    train_metrics = self.train_epoch(epoch)
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/train/trainer.py", line 134, in train_epoch
    outputs = self.model(src, tgt)
              ^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/model/base_transformer.py", line 475, in forward
    encoder_output, enc_attn = self.encoder(src, src_mask, return_attention)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/model/base_transformer.py", line 268, in forward
    x, attn_weights = layer(x, src_mask, return_attention)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/model/base_transformer.py", line 106, in forward
    attn_output, attn_weights = self.self_attn(
                                ^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/miniconda3/envs/ai_hw/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whs/ai_hw/code/model/attention_module.py", line 218, in forward
    K = K * mask.float()
        ~~^~~~~~~~~~~~~~
RuntimeError: The size of tensor a (64) must match the size of tensor b (20) at non-singleton dimension 3
(ai_hw) whs@whs-OMEN:~/ai_hw/code$ 
```