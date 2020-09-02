# -*- coding: utf-8 -*-
# @Time    : 3/29/19 11:03 AM
# @Author  : zhoujun
import torch
from torch import nn
import numpy as np


class PSELoss(nn.Module):
    def __init__(self, Lambda, ratio=3, reduction='mean'):
        """Implement PSE Loss.
        """
        super(PSELoss, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.Lambda = Lambda
        self.ratio = ratio
        self.reduction = reduction

    def forward(self, outputs:torch.Tensor, gt:torch.Tensor, training_masks:torch.Tensor):
        """

        outputs:[batch_size,result_num,w,h]
        labels :[batch_size,result_num,w,h]
        training_masks:[batch_size,w,h]
        """

        ##sn作为gt中 文本框最大的gt图，作为text区域选择
        ##s1 -  sn-1 作为kernel

        texts = outputs[:, -1, :, :]
        kernels = outputs[:, :-1, :, :]
        gt_texts = gt[:, -1, :, :]
        gt_kernels = gt[:, :-1, :, :]

        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        loss_text = self.dice_loss(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = selected_masks.to(outputs.device)
        kernels_num = gt_kernels.size()[1]
        for i in range(kernels_num):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.stack(loss_kernels).mean(0)
        if self.reduction == 'mean':
            loss_text = loss_text.mean()
            loss_kernels = loss_kernels.mean()
        elif self.reduction == 'sum':
            loss_text = loss_text.sum()
            loss_kernels = loss_kernels.sum()

        # loss = self.Lambda * loss_text + (1 - self.Lambda) * loss_kernels
        # 在runner中会将该loss 求和回传
        result = dict(
            loss_text=loss_text * self.Lambda,loss_kernels=(1 - self.Lambda) * loss_kernels
        )
        return result
        # return loss_text, loss_kernels, loss

    def dice_loss(self, input_tensor:torch.Tensor, target:torch.Tensor, mask:torch.Tensor):

        input_tensor = torch.sigmoid(input_tensor)
        input_tensor = input_tensor.contiguous().view(input_tensor.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input_tensor = input_tensor * mask
        target = target * mask

        a = torch.sum(input_tensor * target, 1)
        b = torch.sum(input_tensor * input_tensor, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * 3, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        # 将负样本得分从高到低排序
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks
