from torch.utils.data import Dataset
from .basehook import BaseHook
from ....datasets import build_dataset
import random
import torch
from torch.nn.parallel import DataParallel,DistributedDataParallel
import Polygon as plg
from ....core.evaluation import eval_poly_detect,eval_text
from ....utils.dist_utils import get_dist_info


class DetEvalHook(BaseHook):
    def __init__(self,dataset,
                 batch_size=4,
                 by_epoch=False,
                 interval=5):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            drop_last=True
        )
        self.by_epoch = by_epoch
        self.interval = interval
        # self.rank,self.world_size = get_dist_info()
        # print(self.rank)

    def after_train_epoch(self, runner):
        if  not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        else:
            self.eval(runner)
    def after_train_iter(self, runner):
        if  self.by_epoch or not self.every_n_iters(runner,self.interval):
            return
        else:
            self.eval(runner)


    def eval(self,runner):
        if hasattr(runner.model, "module"):
            device = next(runner.model.module.parameters()).device
        else:
            device = next(runner.model.parameters()).device
        runner.model.eval()
        results = []
        gts = []
        for data in (self.data_loader):
            data['img'] = data['img'].to(device)
            with torch.no_grad():
                result = runner.model(data=data, return_loss=False)
            if type(runner.model) == DataParallel or type(runner.model) == DistributedDataParallel:
                result, scores = runner.model.module.postprocess(result)
            else:
                result, scores = runner.model.postprocess(result)
            results.extend(detect_pred_func(result))
            gt = detect_gt_func(data)
            gts.extend(gt)
        eval_result_dict=eval_poly_detect(results,gts)
        log_str_list = []
        for key,value in eval_result_dict.items():
            log_str_list.append("{}:{}".format(key,value))
        runner.logger.info(",".join(log_str_list))
        runner.model.train()


def detect_pred_func(result):
    batch_polys = []
    for batch_pred in result:
        polys = []
        for bbox in batch_pred:
            poly = plg.Polygon(bbox)
            polys.append(poly)
        batch_polys.append(polys)
    return batch_polys

def detect_gt_func(data:dict):
    gts = data.get("gt_polys")
    return tensor2poly(gts)
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



class RecoEvalHook(BaseHook):
    def __init__(self, dataset,
                 batch_size=4,
                 by_epoch=False,
                 interval=5,
                 show_num=4):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.by_epoch = by_epoch
        self.interval = interval
        self.show_num = show_num


    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        else:
            self.eval(runner)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        else:
            self.eval(runner)

    def eval(self, runner):
        if hasattr(runner.model, "module"):
            device = next(runner.model.module.parameters()).device
        else:
            device = next(runner.model.parameters()).device

        runner.model.eval()
        results = []
        gts = []
        for data in self.data_loader:
            data = batch_dict_data_todevice(data,device)
            with torch.no_grad():
                result = runner.model(data=data, return_loss=False)
            if type(runner.model) == DataParallel or type(runner.model) == DistributedDataParallel:
                result,scores = runner.model.module.postprocess(result)
            else:
                result,scores = runner.model.postprocess(result)
            results.extend(reco_pred_func(result))
            gt = reco_gt_func(data)
            gts.extend(gt)
        eval_result_dict = eval_text(results, gts)
        log_str_list = []
        for key, value in eval_result_dict.items():
            log_str_list.append("{}:{}".format(key, value))
        runner.logger.info(",".join(log_str_list))

        labels = []
        preds = []
        for index,data in enumerate(self.data_loader):
            data = batch_dict_data_todevice(data, device)
            with torch.no_grad():
                result = runner.model(data=data, return_loss=False)
            if type(runner.model) == DataParallel or type(runner.model) == DistributedDataParallel:
                result, scores = runner.model.module.postprocess(result)
            else:
                result, scores = runner.model.postprocess(result)
            preds.extend(reco_pred_func(result))
            gt = reco_gt_func(data)
            labels.extend(gt)
            if len(labels)>self.show_num:
                break
        # show some predicted results
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        predicted_result_log = f'\n{dashed_line}\n{head}\n{dashed_line}\n'
        for i in range(self.show_num):
            gt = labels[i]
            pred = preds[i]
            predicted_result_log += f'{gt:25s} | {pred:25s} |\t{str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}\n'

        runner.logger.info(predicted_result_log)
        runner.model.train()


def reco_pred_func(result):
    return result
def reco_gt_func(data:dict):
    gts = data.get("label")
    return gts

def batch_dict_data_tocuda(data:dict):
    for key,values in data.items():
        if hasattr(values,'cuda'):
            data[key]=values.cuda()
    return data


def batch_dict_data_todevice(data:dict,device):
    for key,values in data.items():
        if hasattr(values,"to"):
            data[key]=values.to(device)
    return data


class DistRecoEvalHook(BaseHook):
    def __init__(self,dataset,interval=2,show_number=5,by_epoch = True,**eval_kwargs):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.show_flag = by_epoch  ##show_flag true->epoch, false->iter
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.show_number = show_number
        self.idx_list = list(range(len(self.dataset)))

    # ##for debug
    def after_train_iter(self, runner):
        if  self.show_flag or not self.every_n_iters(runner,self.interval):
            return
        if runner.rank == 0:
            """
            主机进行测试,其他继续
            """
            runner.model.eval()
            ##TODO:eval 整个valid数据集并计算准确率跟
            random_idx = random.sample(self.idx_list, self.show_number)
            labels = []
            preds = []
            with torch.no_grad():
                for idx in random_idx:
                    data = self.dataset[idx]
                    labels.append(data["label"])
                    # compute output
                    img_tensor = data["img"]
                    img_tensor = img_tensor.unsqueeze(0)
                    data["img"] = img_tensor
                    # 判断模型是运行在cpu上还是GPU上
                    if next(runner.model.parameters()).is_cuda:
                        data = batch_dict_data_tocuda(data)
                    # 如果是数据并行,则只用一块gpu来处理结果,不然默认的dataparallel的gather函数会将字符串类型的返回结果处理成str(map)
                    if type(runner.model) == DataParallel or type(runner.model) == DistributedDataParallel:
                        outputs = runner.model.module(data=data, return_loss=False)
                        ##conver tensor to str
                        outputs = runner.model.module.postprocess(outputs)
                    else:
                        outputs = runner.model(data=data, return_loss=False)
                        outputs = runner.model.module.postprocess(outputs)
                    preds.append(outputs[0])
            # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'\n{dashed_line}\n{head}\n{dashed_line}\n'
            for i in range(len(labels)):
                gt = labels[i]
                pred = preds[i]
                predicted_result_log += f'{gt:25s} | {pred:25s} |\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}\n'

            runner.logger.info(predicted_result_log)
            runner.model.train()

    def after_train_epoch(self, runner):
        if  not self.show_flag or not self.every_n_epochs(runner, self.interval):
            return

        if runner.rank == 0:
            """
            主机进行测试,其他继续
            """
            runner.model.eval()
            ##TODO:eval 整个valid数据集并计算准确率跟
            random_idx = random.sample(self.idx_list, self.show_number)
            labels = []
            preds = []
            with torch.no_grad():
                for idx in random_idx:
                    data = self.dataset[idx]
                    labels.append(data["label"])
                    # compute output
                    img_tensor = data["img"]
                    img_tensor = img_tensor.unsqueeze(0)
                    data["img"] = img_tensor
                    # 判断模型是运行在cpu上还是GPU上
                    if next(runner.model.parameters()).is_cuda:
                        data = batch_dict_data_tocuda(data)
                    # 如果是数据并行,则只用一块gpu来处理结果,不然默认的dataparallel的gather函数会将字符串类型的返回结果处理成str(map)
                    if type(runner.model) == DataParallel or type(runner.model) == DistributedDataParallel:
                        outputs = runner.model.module(data=data, return_loss=False)
                        ##conver tensor to str
                        outputs = runner.model.module.postprocess(outputs)
                    else:
                        outputs = runner.model(data=data, return_loss=False)
                        outputs = runner.model.module.postprocess(outputs)
                    preds.append(outputs[0])
            # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'\n{dashed_line}\n{head}\n{dashed_line}\n'
            for i in range(len(labels)):
                gt = labels[i]
                pred = preds[i]
                predicted_result_log += f'{gt:25s} | {pred:25s} |\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}\n'

            runner.logger.info(predicted_result_log)
            runner.model.train()

# class DistEvalHook(BaseHook):
#
#     def __init__(self, dataset, interval=1, **eval_kwargs):
#         if isinstance(dataset, Dataset):
#             self.dataset = dataset
#         elif isinstance(dataset, dict):
#             self.dataset = build_dataset(dataset, {'test_mode': True})
#         else:
#             raise TypeError(
#                 'dataset must be a Dataset object or a dict, not {}'.format(
#                     type(dataset)))
#         self.interval = interval
#         self.eval_kwargs = eval_kwargs
#
#     def after_train_epoch(self, runner):
#         if not self.every_n_epochs(runner, self.interval):
#             return
#         runner.model.eval()
#         results = [None for _ in range(len(self.dataset))]
#         if runner.rank == 0:
#             prog_bar = mmcv.ProgressBar(len(self.dataset))
#         for idx in range(runner.rank, len(self.dataset), runner.world_size):
#             data = self.dataset[idx]
#             data_gpu = scatter(
#                 collate([data], samples_per_gpu=1),
#                 [torch.cuda.current_device()])[0]
#
#             # compute output
#             with torch.no_grad():
#                 result = runner.model(
#                     return_loss=False, rescale=True, **data_gpu)
#             results[idx] = result
#
#             batch_size = runner.world_size
#             if runner.rank == 0:
#                 for _ in range(batch_size):
#                     prog_bar.update()
#
#         if runner.rank == 0:
#             print('\n')
#             dist.barrier()
#             for i in range(1, runner.world_size):
#                 tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
#                 tmp_results = mmcv.load(tmp_file)
#                 for idx in range(i, len(results), runner.world_size):
#                     results[idx] = tmp_results[idx]
#                 os.remove(tmp_file)
#             self.evaluate(runner, results)
#         else:
#             tmp_file = osp.join(runner.work_dir,
#                                 'temp_{}.pkl'.format(runner.rank))
#             mmcv.dump(results, tmp_file)
#             dist.barrier()
#         dist.barrier()
#
#     def evaluate(self, runner, results):
#         eval_res = self.dataset.evaluate(
#             results, logger=runner.logger, **self.eval_kwargs)
#         for name, val in eval_res.items():
#             runner.log_buffer.output[name] = val
#         runner.log_buffer.ready = True
