import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel,DataParallel
from ..core.train import Runner
from texthub.utils import get_root_logger
from ..core.optimizer import build_optimizer
from ..core.train.Hooks import RecoEvalHook,DistSamplerSeedHook,DistRecoEvalHook,DistOptimizerHook
from torch.utils.data.distributed import DistributedSampler
from texthub.utils.dist_utils import get_dist_info

def train_recoginizer(model,
                   dataset,
                   cfg,
                   distributed=True,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)
    batch_processor = recogition_batch_processor
    # start training
    if distributed:
        _dist_train(
            model,
            dataset,
            cfg,
            batch_processor=batch_processor,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)
    else:
        _non_dist_train(
            model,
            dataset,
            cfg,
            batch_processor = batch_processor,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)


def detect_bact_processor(model,data,train_mode=True):
    #TODO:text detect
    pass


def recogition_batch_processor(model, data, train_mode=True):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """

    if train_mode:
        img_tensor = data["img"]
        labels = data["label"]
        #判断模型是运行在cpu上还是GPU上
        if hasattr(model,"module"):
            if next(model.module.parameters()).is_cuda:
                img_tensor = img_tensor.cuda()
                labels = labels.cuda()
        else:
            if next(model.parameters()).is_cuda:
                img_tensor = img_tensor.cuda()
                labels = labels.cuda()
        losses=model(img_tensor, labels, return_loss=True)
        loss, log_vars = parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return outputs
    else:
        img_tensor = data["img"]

        preds_str = model(img_tensor, None, return_loss=False)
        return dict(
            preds_str=preds_str,ori_label=data["ori_label"]
        )


def _dist_train(model,
                dataset,
                cfg,
                batch_processor,
                validate=False,
                logger=None,
                timestamp=None,
                meta=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    rank, world_size = get_dist_info()
    #TODO: dist dataloader
    # data_loaders = [
    #     build_dataloader(
    #         ds,
    #         cfg.data.imgs_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         dist=True,
    #         seed=cfg.seed) for ds in dataset
    # ]
    # put model on gpus

    """
    在多机多卡情况下分布式训练数据的读取也是一个问题，
    不同的卡读取到的数据应该是不同的。
    dataparallel的做法是直接将batch切分到不同的卡，
    这种方法对于多机来说不可取，因为多机之间直接进行数据传输会严重影响效率。
    于是有了利用sampler确保dataloader只会load到整个数据集的一个特定子集的做法。
    DistributedSampler就是做这件事的。它为每一个子进程划分出一部分数据集，以避免不同进程之间数据重复。
    """
    data_loaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size= cfg.data.imgs_per_gpu,
            pin_memory=True,
            sampler=DistributedSampler(ds,num_replicas=world_size,rank=rank)
            ) for ds in dataset
    ]

    model = DistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # # fp16 setting
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
    #                                          **fp16_cfg)
    # else:
    #     optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(DistRecoEvalHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    batch_processor,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    batch_size = cfg.gpus * cfg.data.imgs_per_gpu
    num_workers = cfg.gpus * cfg.data.workers_per_gpu
    data_loaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        ) for ds in dataset
    ]
    if torch.cuda.is_available() and cfg.gpus!=0:
        # put model on gpus
        model = DataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=None)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    #TODO:fp16 trainning
    # # fp16 setting
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(
    #         **cfg.optimizer_config, **fp16_cfg, distributed=False)
    # else:
    #     optimizer_config = cfg.optimizer_config
    optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    else:
        if hasattr(model, 'module'):
            model.module.init_weights()
        else:
            model.init_weights()
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(RecoEvalHook(val_dataset_cfg, **eval_cfg))


    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)





def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars