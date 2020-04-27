from texthub.utils import Config
import  time
import copy
from texthub.apis import  train_recoginizer
from texthub.datasets import build_dataset
from texthub.modules import build_recognizer
from texthub.utils import  get_root_logger
import os.path as osp
import os
config_file = "./configs/testdatasetconfig.py"
cfg = Config.fromfile(config_file)
cfg.gpus = 1
cfg.resume_from=None
cfg.load_from = None
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
os.makedirs(cfg.work_dir,exist_ok=True)
log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
model = build_recognizer(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
datasets = [build_dataset(cfg.data.train)]

if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))


train_recoginizer(
        model,
        datasets,
        cfg,
        validate=True,
        timestamp=timestamp,
        meta=None
)
    # # init the meta dict to record some important information such as
    # # environment info and seed, which will be logged
    # meta = dict()
    # # # log env info
    # # env_info_dict = collect_env()
    # # env_info = '\n'.join([('{}: {}'.format(k, v))
    # #                       for k, v in env_info_dict.items()])
    # # dash_line = '-' * 60 + '\n'
    # # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    # #             dash_line)
    # # meta['env_info'] = env_info
    #
    # # log some basic info
# logger.info('Config:\n{}'.format(cfg.text))
#
#     # set random seeds
#     if args.seed is not None:
#         logger.info('Set random seed to {}, deterministic: {}'.format(
#             args.seed, args.deterministic))
#         set_random_seed(args.seed, deterministic=args.deterministic)
#     cfg.seed = args.seed
#     meta['seed'] = args.seed
#
#     model = build_recognizer(
#         cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
#
#     datasets = [build_dataset(cfg.data.train)]
#     if len(cfg.workflow) == 2:
#         val_dataset = copy.deepcopy(cfg.data.val)
#         val_dataset.pipeline = cfg.data.train.pipeline
#         datasets.append(build_dataset(val_dataset))
#     if cfg.checkpoint_config is not None:
#         # save mmdet version, config file content and class names in
#         # checkpoints as meta data
#         cfg.checkpoint_config.meta = dict(
#             mmdet_version=__version__,
#             config=cfg.text,
#             CLASSES=datasets[0].CLASSES)
#     # add an attribute for visualization convenience
#     model.CLASSES = datasets[0].CLASSES
#     train_detector(
#         model,
#         datasets,
#         cfg,
#         distributed=distributed,
#         validate=args.validate,
#         timestamp=timestamp,
#         meta=meta)


