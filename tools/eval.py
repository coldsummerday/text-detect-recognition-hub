from __future__ import division
import argparse
import os
import os.path as osp
import sys
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))

import torch
from torch.nn.parallel import DataParallel,DistributedDataParallel

import Polygon as plg

from texthub.utils import Config
from texthub.datasets import  build_dataset
from texthub.utils.processbar import ProgressBar
from texthub.modules import build_recognizer,build_detector
from texthub.core.utils.checkpoint import load_checkpoint
from texthub.core.evaluation import eval_poly_detect,eval_text
def model_inference(model,data_loader,get_gt_func)->([],[]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    dataset = data_loader.dataset
    results = []
    gts = []
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        data['img'] = data['img'].to(device)
        with torch.no_grad():
            result = model(data,return_loss=False)
        if type(model) == DataParallel or type(model) == DistributedDataParallel:
            result = model.module.postprocess(result)
        else:
            result = model.postprocess(result)
        results.extend(result)
        gt = get_gt_func(data)
        gts.extend(gt)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results,gts

def parse_args():
    parser = argparse.ArgumentParser(
        description='textHub test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval',
        choices=['detect', 'reco'],
        default='detect',
        help='model type')
    parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='number of gpus to use ')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True



    dataset = build_dataset(cfg.data.val)

    if args.gpus !=0:
        batch_size = args.gpus * cfg.data.imgs_per_gpu
        num_workers = args.gpus * cfg.data.workers_per_gpu
    else:
        batch_size = cfg.data.imgs_per_gpu
        num_workers = cfg.data.workers_per_gpu
    data_loader =torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )



    # build the model and load checkpoint
    if args.eval=="detect":
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    else:
        model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if torch.cuda.is_available() and args.gpus > 1:
        model = DataParallel(model,device_ids=torch.cuda.current_device())
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    if args.eval == "detect":
        preds,gts = model_inference(model,data_loader,get_gt_func=detect_gt_func)
        print(eval_poly_detect(preds,gts))
    else:
        preds, gts = model_inference(model, data_loader, get_gt_func=reco_gt_func)
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

def detect_gt_func(data:dict):
    gts = data.get("gt_polys")
    return tensor2poly(gts)

def reco_gt_func(data:dict):
    gts = data.get("label")
    return gts





if __name__ == '__main__':
    main()