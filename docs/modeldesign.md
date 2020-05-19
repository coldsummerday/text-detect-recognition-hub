

detector 跟 recognizer中
为了保证DistributedDataParallel dataparalle model,
model(x)中, x的输入一般建议为pytorch中的tensor.(方便自动分割) 所以从dataloader中得到的dict data 
在 batchprocess中取出相应的tensor 进行输入计算.
同时,forward 返回的也是tensor 值, 经过 model.postprocess 转化 为相应的输出值,如预测框,如预测文字
    