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




### 部署 推理：



推理 的代码:
看tests/ test_inferrenceapi.py test_detetcorapi.py 有如何得到文本框跟预测

end2end api:
tools/end2endflaskapi.py 提供了利用flask 部署ocr服务的方式，其中提供了文字检测跟文字识别结合的形式
其只要修改det_config_file,det_checkpoint rec_config_file,rec_checkpoint所在路径即可

```bash
##启动命令：
python3 tools/end2endflaskapi.py


##另开终端测试命令：
curl -F 'image=@/data/zhb/data/receipt/end2end/receipt_2nd_icdr15val/imgs/img_10000.jpg' 127.0.0.1:5000/ocr

```





