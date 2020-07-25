
import numpy as np
from queue import Queue
import cv2
import torch
import torch.nn as nn
import itertools
from ..registry import HEADS
import Polygon as plg
import pyclipper
@HEADS.register_module
class DBHead(nn.Module):
    """
           bias: Whether conv layers have bias or not.
           smooth: If true, use bilinear instead of deconv.
           serial: If true, thresh prediction will combine segmentation result as input.
    """
    def __init__(self,inner_channels = 256,neck_out_channels=256//4,
                k=5,thresh:float=0.3,score_thresh=0.7,min_size=3,
                bias=False,
                smooth=False,
                serial=False,
                max_candidates=1000,
                 *args, **kwargs):
        super(DBHead, self).__init__()
        self.max_candidates = max_candidates
        self.neck_out_channels = neck_out_channels
        self.k = k
        self.thresh = thresh
        self.score_thresh = score_thresh
        self.min_size = min_size
        self.serial = serial
        #对neck融合的特征生成二值化图像
        self.binarize_layer = nn.Sequential(
            nn.Conv2d(inner_channels, self.neck_out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(self.neck_out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.neck_out_channels, self.neck_out_channels, 2, 2),
            nn.BatchNorm2d(self.neck_out_channels),
            nn.ReLU(inplace=True),
            ##输出通道是1，生成一个概率图
            nn.ConvTranspose2d(self.neck_out_channels, 1, 2, 2),
            nn.Sigmoid()
        )
        # use adaptive threshold training
        self.thresh_layer = self._init_thresh_layer(
                inner_channels, serial=serial, smooth=smooth, bias=bias)

        self.loss_fun = L1BalanceCELoss()


    def pred_binarize(self,pred:torch.Tensor):
        return pred>self.thresh

    def forward(self,data:dict,return_loss=False):
        """
        """
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def forward_test(self,data:dict):
        features = data.get("img")
        binary = self.binarize_layer(features)
        return binary

    def forward_train(self,data:dict):
        features = data.get("img")
        binary = self.binarize_layer(features)
        if self.serial:
            features = torch.cat(
                    (features, nn.functional.interpolate(
                        binary, features.shape[2:])), 1)
        thresh = self.thresh_layer(features)
        thresh_binary = self.step_function(binary, thresh)

        gt = data.get("gt")
        mask = data.get("mask")
        thresh_map = data.get("thresh_map")
        thresh_mask = data.get("thresh_mask")
        #caculate loss
        loss_result = self.loss_fun(binary=binary,gt=gt,mask=mask,
                                    thresh = thresh,thresh_binary=thresh_binary,thresh_map=thresh_map,thresh_mask=thresh_mask)
        return loss_result


    def postprocess(self, pred:torch.Tensor,is_output_polygon=True):
        """
        :param binary:
        :return:
        binary: text region segmentation map, with shape (N, 1, H, W)
        process binary to polygon or bbox

        可能需要原始图像的width, height信息
        """
        batches = pred.size(0)

        segmentation = self.pred_binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(batches):
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(
                    pred[batch_index],segmentation[batch_index])
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch,scores_batch

    def polygons_from_bitmap(self,pred:torch.Tensor,segmentation:torch.Tensor):
        '''
                segmentation: single map with shape (1, H, W),
                    whose values are binarized as {0, 1}
        '''
        assert  segmentation.size(0)==1
        segmentation = segmentation.cpu().numpy()[0]
        pred = pred.cpu().detach().numpy()[0]

        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (segmentation * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0]<4:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if score < self.score_thresh:
                continue
            if points.shape[0] < 2:
                continue
            box = self.unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
            box = box.reshape(-1, 2)
            #得到该polygon 的bouding box的小的边长，边长过小的实例丢弃
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))

            if sside < self.min_size + 2:
                continue
            #恢复原图尺寸
            # box[:, 0] = np.clip(
            #     np.round(box[:, 0] / width * dest_width), 0, dest_width)
            # box[:, 1] = np.clip(
            #     np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes,scores

    def unclip(self, box:np.ndarray, unclip_ratio=1.5):
        """
        box :(N,2)
        """
        poly = box.copy().astype(np.int)
        distance = cv2.contourArea(poly) * unclip_ratio/ cv2.arcLength(poly, True)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(-distance))
        return expanded






    def _init_thresh_layer(self,inner_channels,serial=False,smooth=False,bias=False):
        thresh_in_channels = inner_channels
        if serial:
            thresh_in_channels += 1
        thresh_layer = nn.Sequential(
            nn.Conv2d(thresh_in_channels,self.neck_out_channels,kernel_size=3,padding=1,bias=bias),
            nn.BatchNorm2d(self.neck_out_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.neck_out_channels,self.neck_out_channels,smooth=smooth,bias=bias),
            nn.BatchNorm2d(self.neck_out_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.neck_out_channels,1,smooth=smooth,bias=bias),
            nn.Sigmoid()
        )
        return thresh_layer
    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap:np.ndarray, _box:np.ndarray):
        """
        从概率图中得到某个polygon的得分，即统计polygon内bitmap的均值
        bitmap:(H, W)
        _box:(N,2)
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

    def init_weights(self,pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 1e-4)
        else:
            pass



class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween two heatmaps.
    '''
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        assert pred.dim() == 4, pred.dim()
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps

        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss

class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum
        else:
            loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return loss

class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())


        loss = nn.functional.binary_cross_entropy(
            pred, gt, reduction='none')[:, 0, :, :]
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_count = min(int(negative.float().sum()),
                             int(positive_count * self.negative_ratio),negative_loss.nelement())

        negative_loss, _ = torch.topk(negative_loss.view(-1),k= negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class L1BalanceCELoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self,binary:torch.Tensor,
                gt:torch.Tensor,
                mask:torch.Tensor,
                thresh:torch.Tensor,
                thresh_binary:torch.Tensor,
                thresh_map:torch.Tensor,
                thresh_mask:torch.Tensor
                ):
        bce_loss = self.bce_loss(binary, gt, mask)
        thresh_l1_loss = self.l1_loss(thresh, thresh_map, thresh_mask)
        thresh_dice_loss = self.dice_loss(thresh_binary, gt, mask)

        #在runner中会将该loss 求和回传
        result = dict(
            loss_bce=self.bce_scale*bce_loss, loss_thresh_l1=self.l1_scale * thresh_l1_loss,loss_thresh_dice=thresh_dice_loss
        )
        return result

    # def forward(self, pred, batch):
    #     bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
    #     metrics = dict(bce_loss=bce_loss)
    #     if 'thresh' in pred:
    #         l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
    #         dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
    #         metrics['thresh_loss'] = dice_loss
    #         loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
    #         metrics.update(**l1_metric)
    #     else:
    #         loss = bce_loss
    #     return loss, metrics
