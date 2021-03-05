#
# import torch
# from torch.nn.parallel import DistributedDataParallel,DataParallel
#
# from ..core.train import EpochBaseRunner,IterBasedRunner,Runner
# from ..utils import get_root_logger
# from ..core.optimizer import build_optimizer
# from ..core.train.Hooks import RecoEvalHook,DistRecoEvalHook,DistOptimizerHook
# from torch.utils.data.distributed import DistributedSampler
# from ..utils.dist_utils import get_dist_info
# from .batchprocess import recogition_batch_processor,detect_batch_processor
#
#
# def train_detector(model,
#                    dataset,
#                    cfg,
#                    distributed=True,
#                    validate=False,
#                    timestamp=None,
#                    meta=None):
#     logger = get_root_logger(cfg.log_level)
#     batch_processor = detect_batch_processor
#     # start training
#     if distributed:
#         _dist_train(
#             model,
#             dataset,
#             cfg,
#             batch_processor=batch_processor,
#             validate=validate,
#             logger=logger,
#             timestamp=timestamp,
#             meta=meta)
#     else:
#         _non_dist_train(
#             model,
#             dataset,
#             cfg,
#             batch_processor = batch_processor,
#             validate=validate,
#             logger=logger,
#             timestamp=timestamp,
#             meta=meta)
#
#
# def train_recoginizer(model,
#                    dataset,
#                    cfg,
#                    distributed=True,
#                    validate=False,
#                    timestamp=None,
#                    meta=None):
#     logger = get_root_logger(cfg.log_level)
#     batch_processor = recogition_batch_processor
#     # start training
#     if distributed:
#         _dist_train(
#             model,
#             dataset,
#             cfg,
#             batch_processor=batch_processor,
#             validate=validate,
#             logger=logger,
#             timestamp=timestamp,
#             meta=meta)
#     else:
#         _non_dist_train(
#             model,
#             dataset,
#             cfg,
#             batch_processor = batch_processor,
#             validate=validate,
#             logger=logger,
#             timestamp=timestamp,
#             meta=meta)
#
#
# def _dist_train(model,
#                 dataset,
#                 cfg,
#                 batch_processor,
#                 validate=False,
#                 logger=None,
#                 timestamp=None,
#                 meta=None):
#     # prepare data loaders
#     dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
#     rank, world_size = get_dist_info()
#     #TODO: dist dataloader
#     # data_loaders = [
#     #     build_dataloader(
#     #         ds,
#     #         cfg.data.imgs_per_gpu,
#     #         cfg.data.workers_per_gpu,
#     #         dist=True,
#     #         seed=cfg.seed) for ds in dataset
#     # ]
#     # put model on gpus
#
#     """
#     在多机多卡情况下分布式训练数据的读取也是一个问题，
#     不同的卡读取到的数据应该是不同的。
#     dataparallel的做法是直接将batch切分到不同的卡，
#     这种方法对于多机来说不可取，因为多机之间直接进行数据传输会严重影响效率。
#     于是有了利用sampler确保dataloader只会load到整个数据集的一个特定子集的做法。
#     DistributedSampler就是做这件事的。它为每一个子进程划分出一部分数据集，以避免不同进程之间数据重复。
#     """
#     data_loaders = [
#
#         torch.utils.data.DataLoader(
#             ds,
#             batch_size= cfg.data.imgs_per_gpu,
#             pin_memory=True,
#             drop_last=True,
#             sampler=DistributedSampler(ds,num_replicas=world_size,rank=rank)
#             ) for ds in dataset
#     ]
#
#     model = DistributedDataParallel(
#         model.cuda(),
#         device_ids=[torch.cuda.current_device()],
#         broadcast_buffers=False)
#
#     # build runner
#     optimizer = build_optimizer(model, cfg.optimizer)
#
#     iters_num = cfg.get('total_iters', None)
#     if iters_num == None:
#         runner = Runner(
#             model,
#             batch_processor,
#             optimizer,
#             cfg.work_dir,
#             logger=logger,meta=None
#         )
#         # runner = EpochBaseRunner(
#         #     model,
#         #     batch_processor,
#         #     optimizer,
#         #     cfg.work_dir,
#         #     logger=logger,
#         #     meta=None)
#     else:
#         runner = IterBasedRunner(
#             model,
#             batch_processor,
#             optimizer,
#             cfg.work_dir,
#             logger=logger,
#             meta=None)
#
#     # an ugly walkaround to make the .log and .log.json filenames the same
#     runner.timestamp = timestamp
#
#     # # fp16 setting
#     # fp16_cfg = cfg.get('fp16', None)
#     # if fp16_cfg is not None:
#     #     optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
#     #                                          **fp16_cfg)
#     # else:
#     #     optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
#
#     optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
#     # register hooks
#     runner.register_training_hooks(cfg.lr_config, optimizer_config,
#                                    cfg.checkpoint_config, cfg.log_config)
#
#     # register eval hooks
#     if validate:
#         val_dataset_cfg = cfg.data.val
#         eval_cfg = cfg.get('evaluation', {})
#         runner.register_hook(DistRecoEvalHook(val_dataset_cfg, **eval_cfg))
#
#     if cfg.resume_from:
#         runner.resume(cfg.resume_from)
#     elif cfg.load_from:
#         runner.load_checkpoint(cfg.load_from)
#     iters_num = cfg.get('total_iters', None)
#     if iters_num!=None:
#         runner.run(data_loaders, cfg.workflow, iters_num)
#     else:
#         ##epoch base runner 需要， iter base 以及在dataloader设置了
#         # runner.register_hook(DistSamplerSeedHook())
#         runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
#
#
# def _non_dist_train(model,
#                     dataset,
#                     cfg,
#                     batch_processor,
#                     validate=False,
#                     logger=None,
#                     timestamp=None,
#                     meta=None):
#
#     # prepare data loaders
#     dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
#     batch_size = cfg.gpus * cfg.data.imgs_per_gpu
#     num_workers = cfg.gpus * cfg.data.workers_per_gpu
#     data_loaders = [
#         torch.utils.data.DataLoader(
#             ds,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             shuffle=True,
#             pin_memory=True,
#             drop_last=True,
#         ) for ds in dataset
#     ]
#     if torch.cuda.is_available() and cfg.gpus!=0:
#         # put model on gpus
#         model = DataParallel(model, device_ids=range(cfg.gpus)).cuda()
#
#     # build runner
#     optimizer = build_optimizer(model, cfg.optimizer)
#     iters_num = cfg.get('total_iters', None)
#     if iters_num==None:
#         runner = EpochBaseRunner(
#             model,
#             batch_processor,
#             optimizer,
#             cfg.work_dir,
#             logger=logger,
#             meta=None)
#     else:
#         runner = IterBasedRunner(
#             model,
#             batch_processor,
#             optimizer,
#             cfg.work_dir,
#             logger=logger,
#             meta=None)
#
#     # an ugly walkaround to make the .log and .log.json filenames the same
#     runner.timestamp = timestamp
#
#     #TODO:fp16 trainning
#     # # fp16 setting
#     # fp16_cfg = cfg.get('fp16', None)
#     # if fp16_cfg is not None:
#     #     optimizer_config = Fp16OptimizerHook(
#     #         **cfg.optimizer_config, **fp16_cfg, distributed=False)
#     # else:
#     #     optimizer_config = cfg.optimizer_config
#     optimizer_config = cfg.optimizer_config
#     runner.register_training_hooks(cfg.lr_config, optimizer_config,
#                                    cfg.checkpoint_config, cfg.log_config)
#
#     if cfg.resume_from:
#         runner.resume(cfg.resume_from)
#     elif cfg.load_from:
#         runner.load_checkpoint(cfg.load_from)
#     else:
#         if hasattr(model, 'module'):
#             model.module.init_weights()
#         else:
#             model.init_weights()
#     if validate:
#         val_dataset_cfg = cfg.data.val
#         eval_cfg = cfg.get('evaluation', {})
#         runner.register_hook(RecoEvalHook(val_dataset_cfg, **eval_cfg))
#
#     iters_num = cfg.get('total_iters', None)
#     if iters_num != None:
#         runner.run(data_loaders, cfg.workflow, iters_num)
#     else:
#         runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
#
#
#
#
#
