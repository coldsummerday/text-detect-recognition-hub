import numpy as np
from queue import Queue
import cv2
import torch
import torch.nn as nn
import itertools
from ..registry import HEADS
from ...ops.pa import pa_cpp_f
import pyclipper
@HEADS.register_module
class PanCPPHead(nn.Module):
    def __init__(self,min_area = 5,
                 min_score = 0.85,
                 alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean',
                 is_output_polygon = False):
        """

                :param alpha: loss kernel 前面的系数
                :param beta: loss agg 和 loss dis 前面的系数
                :param delta_agg: 计算loss agg时的常量
                :param delta_dis: 计算loss dis时的常量
                :param ohem_ratio: OHEM的比例
                :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
                """
        super().__init__()
        self.loss = PANLoss(alpha=alpha, beta=beta, delta_agg=delta_agg, delta_dis=delta_dis, ohem_ratio=ohem_ratio, reduction=reduction)
        self.min_area = min_area
        self.min_score = min_score
        self.is_output_polygon = is_output_polygon
    def forward(self,data:dict,return_loss=False):
        """
        forward do nothing,
        train calcalate loss in panloss
        inference preds in postprocess to get the predbboxes
        :param img_tensor:
        :param extra_data:
        :param return_loss:
        :return:
        """
        if return_loss:
            img = data.get("img")
            loss_dict = self.loss(outputs=img, labels=data.get('gt'), training_masks=data.get('mask'))
            return loss_dict
        else:
            return data['img']

    def postprocess(self,preds):
        """

        :param preds:
        :return:
        """
        batch_bbox_list = []
        score_list = []
        for batch_preds in preds:
            boxes_list,score_array = self.get_results(batch_preds)

            batch_bbox_list.append(boxes_list)
            score_list.append(score_array)
        return batch_bbox_list,score_list #bbox,scores

    def init_weights(self, pretrained=None):
        pass


    def get_results(self,pred:torch.Tensor):
        """
        preds (6,h,w)
        [:,0,:,:] pred_text_pixel
        [:,1,:,:] textkernel
        [:,2:,:,:] similarity_vectors
        """

        score = torch.sigmoid(pred[0,:,:])
        kernels = pred[:2,:,:] > 0
        text_mask = kernels[:1,:,:]

        ##kernel * text 保证kernel 肯定在text实例里面
        kernels[1:,:,:] = kernels[1:,:,:] * text_mask
        similarity_vectors = pred[2:,:,:] * text_mask.float()

        score = score.data.cpu().numpy().astype(np.float32)
        kernels = kernels.data.cpu().numpy().astype(np.uint8)
        similarity_vectors = similarity_vectors.cpu().numpy().astype(np.float32)

        # import time
        # start_time = time.time()
        # label = _pa_warpper(kernels, similarity_vectors, min_area=self.min_area)
        # end_time = time.time()
        # print("pa post_process spend time",end_time-start_time)

        label = _pa_warpper(kernels, similarity_vectors, min_area=self.min_area)

        label_num = np.max(label) + 1

        bboxes = []
        scores = []

        for i in range(1,label_num):
            ind = label==i
            points = np.array(np.where(ind)).transpose((1,0))
            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue
            score_i = np.mean(score[ind])
            if score_i < self.min_score:
                label[ind] = 0
                continue
            if  self.is_output_polygon:
                raise NotImplementedError
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                contour = contours[0]
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                bbox = approx.reshape((-1, 2)).astype(np.int32)
                if bbox.shape[0] < 4:
                    continue
                bboxes.append(bbox)
            else:
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect)
                bboxes.append([bbox[1], bbox[2], bbox[3], bbox[0]])
            scores.append(score_i)
        if not self.is_output_polygon:
            bboxes = np.array(bboxes)
        return bboxes,scores

    def unclip(self, box:np.ndarray, unclip_ratio=1.5):
        """
        box :(N,2)
        The Clipper library uses integers instead of floating point values to preserve numerical robustness
        """
        poly = box.copy().astype(np.int)
        distance = cv2.contourArea(poly) * unclip_ratio/ cv2.arcLength(poly, True)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # expanded = np.array(offset.Execute(distance)) #offset.Execute(distance)有时候会有两个结果
        polygon_expanded = offset.Execute(distance)
        ##收缩放大后可能产生多个区域，选择点个数最多的那个多边形（暂时最大
        if len(polygon_expanded)>1:
            max_len= 0
            max_polygon =None
            for expanded_poly in polygon_expanded:
                if len(expanded_poly) > max_len:
                    max_len = len(expanded_poly)
                    max_polygon = expanded_poly
            expanded = np.array(max_polygon)
        else:
            expanded = np.array(polygon_expanded)
        return expanded


# def pa_warpper(kernels:np.ndarray,similarity_vectors:np.ndarray,min_area = 2)->np.ndarray:
#     """
#     kernels:(2,h,w)
#     similarity_vectors:(4,h,w)
#
#     return (h,w)
#     """
#     ##文字区域
#     _,cc = cv2.connectedComponents(kernels[0],connectivity=4)
#     ##kernel 最小连通区域,从label出发将cc中的像素进行聚类
#     label_num,label = cv2.connectedComponents(kernels[1],connectivity=4)
#
#
#
#     return pa_cpp_f(kernels[-1], similarity_vectors, label, cc, label_num, min_area)

def _pa_warpper(kernels: np.ndarray, similarity_vectors: np.ndarray, min_area=2) -> np.ndarray:
    """
    kernels:(2,h,w)
    similarity_vectors:(4,h,w)

    return (h,w)
    """
    ##文字区域
    _, cc = cv2.connectedComponents(kernels[0], connectivity=4)
    ##kernel 最小连通区域,从label出发将cc中的像素进行聚类
    label_num, label = cv2.connectedComponents(kernels[1], connectivity=4)
    return pa_cpp_f(similarity_vectors, label, cc, label_num, min_area)




class PANLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean'):
        """
        Implement PAN Loss.
        :param alpha: loss kernel 前面的系数
        :param beta: loss agg 和 loss dis 前面的系数
        :param delta_agg: 计算loss agg时的常量
        :param delta_dis: 计算loss dis时的常量
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.delta_agg = delta_agg
        self.delta_dis = delta_dis
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, outputs, labels, training_masks):
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1, :, :]
        gt_texts = labels[:, 0, :, :]
        gt_kernels = labels[:, 1, :, :]


        # 计算 agg loss 和 dis loss
        similarity_vectors = outputs[:, 2:, :, :]
        loss_aggs, loss_diss = self.agg_dis_loss(texts, kernels, gt_texts, gt_kernels, similarity_vectors)

        # 计算 text loss
        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        loss_texts = self.dice_loss(texts, gt_texts, selected_masks)

        # 计算 kernel loss
        # selected_masks = ((gt_texts > 0.5) & (training_masks > 0.5)).float()
        mask0 = torch.sigmoid(texts).detach().cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float().to(texts.device)
        loss_kernels = self.dice_loss(kernels, gt_kernels, selected_masks)

        # mean or sum
        if self.reduction == 'mean':
            loss_text = loss_texts.mean()
            loss_kernel = loss_kernels.mean()
            loss_agg = loss_aggs.mean()
            loss_dis = loss_diss.mean()
        elif self.reduction == 'sum':
            loss_text = loss_texts.sum()
            loss_kernel = loss_kernels.sum()
            loss_agg = loss_aggs.sum()
            loss_dis = loss_diss.sum()
        else:
            raise NotImplementedError

        #loss_all = loss_text + self.alpha * loss_kernel + self.beta * (loss_agg + loss_dis)
        # result = dict(
        #     loss_text=loss_text, loss_kernel=self.alpha*loss_kernel, loss_agg=self.beta*loss_agg, loss_dis=self.beta*loss_dis
        # )
        result = dict(
            loss_text=loss_text, loss_kernel=loss_kernel, loss_agg=loss_agg, loss_dis=loss_dis,
            loss = loss_text + self.alpha * loss_kernel + self.beta * (loss_agg + loss_dis),
        )

        return result

    def agg_dis_loss(self, texts, kernels, gt_texts, gt_kernels, similarity_vectors):
        """
        计算 loss agg
        :param texts: 文本实例的分割结果 batch_size * (w*h)
        :param kernels: 缩小的文本实例的分割结果 batch_size * (w*h)
        :param gt_texts: 文本实例的gt batch_size * (w*h)
        :param gt_kernels: 缩小的文本实例的gt batch_size*(w*h)
        :param similarity_vectors: 相似度向量的分割结果 batch_size * 4 *(w*h)
        :return:
        """
        batch_size = texts.size()[0]
        texts = texts.contiguous().reshape(batch_size, -1)
        kernels = kernels.contiguous().reshape(batch_size, -1)
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)
        similarity_vectors = similarity_vectors.contiguous().view(batch_size, 4, -1)
        loss_aggs = []
        loss_diss = []
        for text_i, kernel_i, gt_text_i, gt_kernel_i, similarity_vector in zip(texts, kernels, gt_texts, gt_kernels,
                                                                               similarity_vectors):
            text_num = gt_text_i.max().item() + 1
            loss_agg_single_sample = []
            G_kernel_list = []  # 存储计算好的G_Ki,用于计算loss dis
            # 求解每一个文本实例的loss agg
            for text_idx in range(1, int(text_num)):
                # 计算 D_p_Ki
                single_kernel_mask = gt_kernel_i == text_idx
                if single_kernel_mask.sum() == 0 or (gt_text_i == text_idx).sum() == 0:
                    # 这个文本被crop掉了
                    continue
                # G_Ki, shape: 4
                G_kernel = similarity_vector[:, single_kernel_mask].mean(1)  # 4
                G_kernel_list.append(G_kernel)
                # 文本像素的矩阵 F(p) shape: 4* nums (num of text pixel)
                text_similarity_vector = similarity_vector[:, gt_text_i == text_idx]
                # ||F(p) - G(K_i)|| - delta_agg, shape: nums
                text_G_ki = (text_similarity_vector - G_kernel.reshape(4, 1)).norm(2, dim=0) - self.delta_agg
                # D(p,K_i), shape: nums
                D_text_kernel = torch.max(text_G_ki, torch.tensor(0, device=text_G_ki.device, dtype=torch.float)).pow(2)
                # 计算单个文本实例的loss, shape: nums
                loss_agg_single_text = torch.log(D_text_kernel + 1).mean()
                loss_agg_single_sample.append(loss_agg_single_text)
            if len(loss_agg_single_sample) > 0:
                loss_agg_single_sample = torch.stack(loss_agg_single_sample).mean()
            else:
                loss_agg_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_aggs.append(loss_agg_single_sample)

            # 求解每一个文本实例的loss dis
            loss_dis_single_sample = 0
            for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
                # delta_dis - ||G(K_i) - G(K_j)||
                kernel_ij = self.delta_dis - (G_kernel_i - G_kernel_j).norm(2)
                # D(K_i,K_j)
                D_kernel_ij = torch.max(kernel_ij, torch.tensor(0, device=kernel_ij.device, dtype=torch.float)).pow(2)
                loss_dis_single_sample += torch.log(D_kernel_ij + 1)
            if len(G_kernel_list) > 1:
                loss_dis_single_sample /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
            else:
                loss_dis_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_diss.append(loss_dis_single_sample)
        return torch.stack(loss_aggs), torch.stack(loss_diss)

    def dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
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
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
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



