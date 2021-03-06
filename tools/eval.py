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
        description='textHub test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--eval',
        choices=['detect', 'reco'],
        default='detect',
        help='model type')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True

    # set random seeds
    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=args.deterministic)



    dataset = build_dataset(cfg.data.val)


    batch_size = cfg.data.batch_size
    data_loader =torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )



    # build the model and load checkpoint
    if args.eval=="detect":
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    else:
        model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =model.to(device)
    # 单卡好像有bug
    # if torch.cuda.is_available() and args.gpus > 1:
    #     model = DataParallel(model,device_ids=range(args.gpus)).cuda()
    # else:
    #     if torch.cuda.is_available():
    #         model = model.cuda()

    if args.eval == "detect":
        preds,gts = model_inference(model,data_loader,get_pred_func=detect_pred_func,get_gt_func=detect_gt_func)
        print(eval_poly_detect(preds,gts))
    else:
        preds, gts = model_inference(model, data_loader, get_pred_func=reco_pred_func,get_gt_func=reco_gt_func)
        print(eval_text(preds,gts))


def tensor2poly(gt_polys:torch.Tensor):
    #(b,150,4,2)
    results = []
    for array in gt_polys:
        image_polys = []
        for points in array:
            if points[0,0]!=0:
                poly_gon = plg.Polygon(points.cpu().numpy())
                image_polys.append(poly_gon)
        results.append(image_polys)
    return results


def detect_pred_func(result):
    batch_polys = []
    for batch_pred in result:
        polys = []
        for bbox in batch_pred:
            poly = plg.Polygon(bbox)
            polys.append(poly)
        batch_polys.append(polys)
    return batch_polys

def reco_pred_func(result):
    return result



def detect_gt_func(data:dict):
    gts = data.get("gt_polys")
    return tensor2poly(gts)

def reco_gt_func(data:dict):
    gts = data.get("label")
    return gts





if __name__ == '__main__':
    main()