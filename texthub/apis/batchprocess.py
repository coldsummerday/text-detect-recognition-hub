import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
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
        if hasattr(model,"module"):
            if next(model.module.parameters()).is_cuda:
                img_tensor = img_tensor.cuda()
        else:
            if next(model.parameters()).is_cuda:
                img_tensor = img_tensor.cuda()
        preds_str = model(img_tensor, None, return_loss=False)
        return dict(
            preds_str=preds_str
        )



def batch_dict_data_tocuda(data:dict):
    for key,values in data.items():
        if hasattr(values,'cuda'):
            data[key]=values.cuda()
    return data




def detect_batch_processor(model, data, train_mode=True):
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
        #判断模型是运行在cpu上还是GPU上
        if hasattr(model,"module"):
            if next(model.module.parameters()).is_cuda:
                data = batch_dict_data_tocuda(data)
        else:
            if next(model.parameters()).is_cuda:
                data = batch_dict_data_tocuda(data)
        losses=model(data,return_loss=True)
        loss, log_vars = parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return outputs
    else:

        if hasattr(model,"module"):
            if next(model.module.parameters()).is_cuda:
                data = batch_dict_data_tocuda(data)
        else:
            if next(model.parameters()).is_cuda:
                data = batch_dict_data_tocuda(data)
        preds = model(data,return_loss=False)
        return preds


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