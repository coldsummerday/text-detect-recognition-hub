from __future__ import division
import argparse
import os
import os.path as osp
import sys
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from tqdm import tqdm
from typing import Callable


import torch
from torch.nn.parallel import DataParallel,DistributedDataParallel

import Polygon as plg

from texthub.utils import Config,set_random_seed
from texthub.datasets import  build_dataset
from texthub.modules import build_recognizer,build_detector
from texthub.core.utils.checkpoint import load_checkpoint
from texthub.core.evaluation import eval_poly_detect,eval_text
import copy
import multiprocessing
from texthub.utils import get_root_logger


def eval_all_benchmark():
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    args = parse_args()
    root_dir = args.dir
    cfg = Config.fromfile(args.config)


    # set random seeds
    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=False)

    base_data_set_cfg = cfg.data.val
    batch_size = cfg.data.batch_size


    # build the model and load checkpoint
    model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    use_cpu_workers = int(multiprocessing.cpu_count() / 4)
    logger = get_root_logger(log_file=None)
    final_dataset_result_dict = {}
    for name in eval_data_list:
        dataset_dir = os.path.join(root_dir,name)
        dataset_cfg = copy.deepcopy(base_data_set_cfg)
        if dataset_cfg["type"]=='ConcateLmdbDataset':
            dataset_dir = [dataset_dir]

        dataset_cfg["root"]=dataset_dir
        dataset = build_dataset(dataset_cfg)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=use_cpu_workers,
            pin_memory=True,
            drop_last=True,
        )
        try:
            logger.info("start eval {} dataset".format(name))
            preds, gts = model_inference(model, data_loader, get_pred_func=reco_pred_func, get_gt_func=reco_gt_func)

            result_dict = eval_text(preds, gts)
            final_dataset_result_dict[name]=result_dict
            logger.info("{} result is:{}".format(name,result_dict))
        except Exception as e:
            logger.error("{}".format(e))
            continue

    for key,value in final_dataset_result_dict.items():
        logger.info("{} result:{}".format(key,value))






def model_inference(model,data_loader,get_pred_func:Callable,get_gt_func:Callable)->([],[]):
    if hasattr(model,"module"):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device
    model.eval()
    results = []
    gts = []
    for  data in tqdm(data_loader):
        data = batch_dict_data_todevice(data, device)
        with torch.no_grad():
            result = model(data=data,return_loss=False)
        if type(model) == DataParallel or type(model) == DistributedDataParallel:
            result,scores = model.module.postprocess(result)
        else:
            result,scores = model.postprocess(result)
        results.extend(get_pred_func(result))

        gt = get_gt_func(data)
        gts.extend(gt)
    return results,gts

def batch_dict_data_todevice(data:dict,device):
    for key,values in data.items():
        if hasattr(values,"to"):
            data[key]=values.to(device)
    return data

def parse_args():
    parser = argparse.ArgumentParser(
        description='textHub eval a model with a ')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument(
        '--dir',
        help='the eval lmdb path')

    args = parser.parse_args()
    return args

def main():
    eval_all_benchmark()





def reco_pred_func(result):
    return result

def reco_gt_func(data:dict):
    gts = data.get("label")
    return gts





if __name__ == '__main__':
    main()