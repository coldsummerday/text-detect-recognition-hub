
from ...ops.pse import get_points,pse_cpp,get_num
##ops.pse为cpp实现方式
import numpy as np
from queue import Queue
import cv2
import torch
import torch.nn as nn
import itertools
from ..registry import HEADS
import Polygon as plg

@HEADS.register_module
class PanHead(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean'):
        """
                Implement PSE Loss.
                :param alpha: loss kernel 前面的系数
                :param beta: loss agg 和 loss dis 前面的系数
                :param delta_agg: 计算loss agg时的常量
                :param delta_dis: 计算loss dis时的常量
                :param ohem_ratio: OHEM的比例
                :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
                """
        super().__init__()
        self.loss = PANLoss(alpha=alpha, beta=beta, delta_agg=delta_agg, delta_dis=delta_dis, ohem_ratio=ohem_ratio, reduction=reduction)

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
        for batch_preds in preds:
            _, boxes_list = decode(batch_preds)
            batch_bbox_list.append(boxes_list)
        return batch_bbox_list,[] #bbox,scores

    def init_weights(self, pretrained=None):
        pass








class PANLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean'):
        """
        Implement PSE Loss.
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
        result = dict(
            loss_text=loss_text, loss_kernel=self.alpha*loss_kernel, loss_agg=self.beta*loss_agg, loss_dis=self.beta*loss_dis
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




def decode_ploy(preds, scale=1, threshold=0.7311, min_area=5):
    """
        在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
        :param preds: 网络输出
        :param scale: 网络的scale
        :param threshold: sigmoid的阈值
        :return: 文本框的ploy
    """
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    score = preds[0].astype(np.float32)
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel
    similarity_vectors = preds[2:].transpose((1, 2, 0))

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_values = []
    label_sum = get_num(label, label_num)
    for label_idx in range(1, label_num):
        if label_sum[label_idx] < min_area:
            continue
        label_values.append(label_idx)

    pred = pse_cpp(text.astype(np.uint8), similarity_vectors, label, label_num, 0.8)
    pred = pred.reshape(text.shape)

    label_points = get_points(pred, score, label_num)
    result = []
    for label_value, label_point in label_points.items():
        if label_value not in label_values:
            continue
        score_i = label_point[0]
        label_point = label_point[2:]
        points = np.array(label_point, dtype=int).reshape(-1, 2)

        if points.shape[0] < 100 / (scale * scale):
            continue

        if score_i < 0.93:
            continue
        points = mask_points_to_contours_points(points)
        result.append(plg.Polygon(points))
    return result
    #     rect = cv2.minAreaRect(points)
    #     bbox = cv2.boxPoints(rect)
    #     bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
    # # preds是返回的mask
    # return pred, np.array(bbox_list)

def mask_points_to_contours_points(points):
    ##从mask中找最大临接矩阵
    max_value = np.max(points)
    mask = np.zeros((max_value + 1, max_value + 1), dtype='uint8')
    putPoints(mask, points)
    # (16,325)的点,如果不转置的话会变(325,16) 原因cv2图的存储方式不是w,h,而是 h,w
    contours, hierarchy = cv2.findContours(mask.T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = contours[0].reshape(-1, 2)
    return points

def putPoints(mask, points):
    ##不能直接mask[points]=1 ,会导致整行整列都为1
    for point in points:
        x, y = point
        mask[x, y] = 1



def decode_bbox(preds, scale=1, threshold=0.7311, min_area=5):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    score = preds[0].astype(np.float32)
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel
    similarity_vectors = preds[2:].transpose((1, 2, 0))

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_values = []
    label_sum = get_num(label, label_num)
    for label_idx in range(1, label_num):
        if label_sum[label_idx] < min_area:
            continue
        label_values.append(label_idx)

    pred = pse_cpp(text.astype(np.uint8), similarity_vectors, label, label_num, 0.8)
    pred = pred.reshape(text.shape)

    bbox_list = []
    label_points = get_points(pred, score, label_num)

    for label_value, label_point in label_points.items():
        if label_value not in label_values:
            continue
        score_i = label_point[0]
        label_point = label_point[2:]
        points = np.array(label_point, dtype=int).reshape(-1, 2)

        if points.shape[0] < 100 / (scale * scale):
            continue

        if score_i < 0.93:
            continue
        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect)
        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
    # preds是返回的mask
    return np.array(bbox_list)



def decode(preds, scale=1, threshold=0.7311, min_area=5):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    score = preds[0].astype(np.float32)
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel
    similarity_vectors = preds[2:].transpose((1, 2, 0))

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_values = []
    label_sum = get_num(label, label_num)
    for label_idx in range(1, label_num):
        if label_sum[label_idx] < min_area:
            continue
        label_values.append(label_idx)

    pred = pse_cpp(text.astype(np.uint8), similarity_vectors, label, label_num, 0.8)
    pred = pred.reshape(text.shape)

    bbox_list = []
    label_points = get_points(pred, score, label_num)

    for label_value, label_point in label_points.items():
        if label_value not in label_values:
            continue
        score_i = label_point[0]
        label_point = label_point[2:]
        points = np.array(label_point, dtype=int).reshape(-1, 2)

        if points.shape[0] < 100 / (scale * scale):
            continue

        if score_i < 0.93:
            continue
        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect)
        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
    # preds是返回的mask
    return pred, np.array(bbox_list)


def decode_dice(preds, scale=1, threshold=0.7311, min_area=5):
    import pyclipper
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label_num == label_idx)).transpose((1, 0))[:, ::-1]

        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)

        d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))

        if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
            continue

        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)


"""
pse的py实现方式
"""
def get_dis(sv1, sv2):
    return np.linalg.norm(sv1 - sv2)
def pse_py(text, similarity_vectors, label, label_values, dis_threshold=0.8):
    pred = np.zeros(text.shape)
    queue = Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))

    for point_idx in range(points.shape[0]):
        y, x = points[point_idx, 0], points[point_idx, 1]
        label_value = label[y, x]
        queue.put((y, x, label_value))
        pred[y, x] = label_value
    # 计算kernel的值
    d = {}
    for i in label_values:
        kernel_idx = label == i
        kernel_similarity_vector = similarity_vectors[kernel_idx].mean(0)  # 4
        d[i] = kernel_similarity_vector

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    kernal = text.copy()
    while not queue.empty():
        (y, x, label_value) = queue.get()
        cur_kernel_sv = d[label_value]
        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpy >= kernal.shape[0] or tmpy < 0 or tmpx >= kernal.shape[1]:
                continue
            if kernal[tmpy, tmpx] == 0 or pred[tmpy, tmpx] > 0:
                continue
            if np.linalg.norm(similarity_vectors[tmpy, tmpx] - cur_kernel_sv) >= dis_threshold:
                continue
            queue.put((tmpy, tmpx, label_value))
            pred[tmpy, tmpx] = label_value
    return pred






