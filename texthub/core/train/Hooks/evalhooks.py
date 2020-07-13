from torch.utils.data import Dataset
from .basehook import BaseHook
from ....datasets import build_dataset
import random
import torch
from torch.nn.parallel import DataParallel,DistributedDataParallel

class RecoEvalHook(BaseHook):
    def __init__(self,dataset,interval=2,show_number=5,**eval_kwargs):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.show_number = show_number
        self.idx_list = list(range(len(self.dataset)))


    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

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

def batch_dict_data_tocuda(data:dict):
    for key,values in data.items():
        if hasattr(values,'cuda'):
            data[key]=values.cuda()
    return data


class DistRecoEvalHook(BaseHook):
    def __init__(self,dataset,interval=2,show_number=5,**eval_kwargs):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.show_number = show_number
        self.idx_list = list(range(len(self.dataset)))

    # ##for debug
    # def after_train_iter(self, runner):
    #
    #     if runner.rank == 0:
    #         """
    #         主机进行测试,其他继续
    #         """
    #         runner.model.eval()
    #         ##TODO:eval 整个valid数据集并计算准确率跟
    #         random_idx = random.sample(self.idx_list, self.show_number)
    #         labels = []
    #         preds = []
    #         with torch.no_grad():
    #             for idx in random_idx:
    #                 data = self.dataset[idx]
    #                 labels.append(data["label"])
    #                 # compute output
    #                 img_tensor = data["img"]
    #                 img_tensor = img_tensor.unsqueeze(0)
    #                 data["img"] = img_tensor
    #                 # 判断模型是运行在cpu上还是GPU上
    #                 if next(runner.model.parameters()).is_cuda:
    #                     data = batch_dict_data_tocuda(data)
    #                 # 如果是数据并行,则只用一块gpu来处理结果,不然默认的dataparallel的gather函数会将字符串类型的返回结果处理成str(map)
    #                 if type(runner.model) == DataParallel or type(runner.model) == DistributedDataParallel:
    #                     outputs = runner.model.module(data=data, return_loss=False)
    #                     ##conver tensor to str
    #                     outputs = runner.model.module.postprocess(outputs)
    #                 else:
    #                     outputs = runner.model(data=data, return_loss=False)
    #                     outputs = runner.model.module.postprocess(outputs)
    #                 preds.append(outputs[0])
    #         # show some predicted results
    #         dashed_line = '-' * 80
    #         head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
    #         predicted_result_log = f'\n{dashed_line}\n{head}\n{dashed_line}\n'
    #         for i in range(len(labels)):
    #             gt = labels[i]
    #             pred = preds[i]
    #             predicted_result_log += f'{gt:25s} | {pred:25s} |\t{str(pred == gt)}\n'
    #             predicted_result_log += f'{dashed_line}\n'
    #
    #         runner.logger.info(predicted_result_log)
    #         runner.model.train()

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
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
