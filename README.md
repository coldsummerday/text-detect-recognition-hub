# text-detect-recognition-hub





bug:detect polygon iou cal Unstable


第一次安装该代码要安装cpp拓展部分.
```python
sudo python3 setup.py build_ext --inplace
```


训练代码:
```python
python3 -m torch.distributed.launch  --nproc_per_node=2  tools/train_*.py  config.py   --gpus 0 
```
训练的参数有保存工作目录, 是否从断点开始恢复等,详情看tools/train.sh


推理 的代码:
看tests/ test_inferrenceapi.py test_detetcorapi.py 有如何得到文本框跟预测


