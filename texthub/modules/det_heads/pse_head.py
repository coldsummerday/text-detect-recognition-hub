import numpy as np
import cv2
import torch
import torch.nn as nn
from ..registry import HEADS
from ..losses import PSELoss
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ...ops.pse import pse_cpp_f
@HEADS.register_module
class PseHead(nn.Module):
    def __init__(self,conv_out=256,result_num=6,ori_w=640,ori_h=640,scale = 1,Lambda = 0.7,OHEM_ratio = 3,
                 pred_threshold = 0.7311,pred_score = 0.93,
                 *args, **kwargs):
        super(PseHead, self).__init__()
        self.img_h = ori_h
        self.img_w = ori_w
        self.scale = scale
        self.threshold = pred_threshold
        self.pred_score = pred_score
        self.conv = nn.Sequential(
            nn.Conv2d(conv_out * 4, conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)
        self.loss_f = PSELoss(Lambda=Lambda,ratio=OHEM_ratio,reduction="mean")


    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)


    def forward(self, data: dict, return_loss=False):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def forward_test(self, data: dict):
        x = data.get("img")
        x = self.conv(x)
        x = self.out_conv(x)
        x = F.interpolate(x, size=(self.img_h // self.scale, self.img_w // self.scale), mode='bilinear', align_corners=True)
        #[batch,result_num,w,h]



        return x

    def forward_train(self,data:dict):
        x = data.get("img")
        x = self.conv(x)
        x = self.out_conv(x)
        x = F.interpolate(x, size=(self.img_h, self.img_w), mode='bilinear', align_corners=True)

        gt = data.get("gt")
        mask = data.get("mask")
        loss_result = self.loss_f(outputs=x, gt=gt, training_masks=mask)
        return loss_result

    def postprocess(self,preds:torch.Tensor)->([],[]):
        batch_size =  preds.shape[0]
        total_bbox_list = []
        total_score_list = []
        for i in range(batch_size):
            bbox_list,score_list = self._post_process_single(preds[i])
            total_bbox_list.append(bbox_list)
            total_score_list.append(score_list)
        return total_bbox_list,total_score_list

    def _post_process_single(self,preds:torch.Tensor):
        """
        preds:[result_num,w,h]
        """
        preds = torch.sigmoid(preds)
        preds = preds.detach().cpu().numpy()


        score = preds[-1].astype(np.float32)
        preds = preds > self.threshold

        # show_img(preds)

        pse_pred, label_values = pse_warpper(preds, 5)
        polygon_list = []
        score_list = []
        for label_value in label_values:
            points = np.array(np.where(pse_pred == label_value)).transpose((1, 0))[:, ::-1]

            # if points.shape[0] < 800 / (1 * 1):
            #     continue

            score_i = np.mean(score[pse_pred == label_value])
            if score_i < self.pred_score:
                continue

            score_list.append(score_i)
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)
            polygon_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
            # polygon_list.append(rect)
        return np.array(polygon_list), score_list


def pse_warpper(kernels,min_area=5):
    '''
        reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
        :param kernels:
        :param min_area:
        :return:
    '''
    kernel_num = len(kernels)
    if not kernel_num or kernel_num==0:
        return np.array([]), []
    kernels = np.array(kernels)

    #从最小的图 先找到每一个文字实例的最小部分
    label_num, label = cv2.connectedComponents(kernels[0].astype(np.uint8), connectivity=4)
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)

    preds = pse_cpp_f(label, kernels, c=kernel_num)

    return np.array(preds), label_values


# def show_img(imgs: np.ndarray, color=False):
#     import matplotlib.pyplot as plt
#     if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
#         imgs = np.expand_dims(imgs, axis=0)
#     for img in imgs:
#         plt.figure()
#         plt.imshow(img, cmap=None if color else 'gray')






