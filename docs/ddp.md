








开启ddp 后出现的问题:

TypeError: can't pickle Environment objects:(lmdb bug)

(1)初始化时没有指定运行多进程运行方式:
在代码中加入这条
```
torch.multiprocessing.set_start_method('spawn')
```
(2)dataloader中num_workers设置
因为ddp是多进程方式,所以 num_workers 要么就用默认的,要么设为0
(3)dataloader中shuffle 要设置为false(多进程dataloader没法shuffle)