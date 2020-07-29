from ..registry import PIPELINES
import  numpy as np

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
# ABCs from collections will be deprecated in python 3.8+,
# while collections.abc is not available in python 2.7
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc
import cv2



@PIPELINES.register_module
class ResizeRecognitionImageCV2(object):
    """
    文本识别的resize
    img_scale:(h,w)
    尽量不resize 图片，只将原始图片贴到mask上，保持图像原来清晰度
    """
    def __init__(self,img_scale=None,img_channel = 1):
        assert  isinstance(img_scale,tuple) and len(img_scale)==2,"img_scale must be tuple(h,w)"
        self.default_h,self.default_w = img_scale
        self.img_channel = img_channel


    def __call__(self, data:{}):
        #应该添加竖排文字图像的处理
        # img
        img = data["img"]
        if img.ndim==2:
            h,w = img.shape
        else:
            h,w,_ = img.shape
        if self.img_channel == 1 and img.ndim==3:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        if h<self.default_h and w < self.default_w:
            if self.img_channel==1:
                masked_image = self.mask_image_gray(img)
            else:
                masked_image = self.mask_image_color(img)
            data["img"] = masked_image
            return data

        ##保持 原始图像长宽比
        if h > self.default_h and w <self.default_w:
            img = self.resize_by_h(img)
        elif w > self.default_w and h<self.default_h:
            img= self.resize_by_w(img)
        else:
            ##宽跟高都超过了限定
            img = cv2.resize(img, (self.default_w, self.default_h))
        if self.img_channel == 1:
            masked_image = self.mask_image_gray(img)
        else:
            masked_image = self.mask_image_color(img)
        data["img"] = masked_image
        return data

        # w, h = img.size
        # # if w < self.default_h and h <self.default_w:
        # #     ##宽跟高都小于reszie  尺寸，为了保持图像文字清晰，应该只做padding操作
        # #
        #
        # ratio = w / float(h)
        # if math.ceil(self.default_h * ratio) > self.default_w:
        #     resized_w = self.default_w
        # else:
        #     resized_w = math.ceil(self.default_h * ratio)
        # resized_image = img.resize((resized_w, self.default_h), Image.BICUBIC)
        # data["img"] = resized_image
        # return data

    def resize_by_h(self,img:np.ndarray):
        if img.ndim==2:
            h,w = img.shape
        else:
            h,w,_ = img.shape
        shrinking_ratio = self.default_h / float(h)
        img = cv2.resize(img, (int(w * shrinking_ratio), self.default_h))
        return img

    def resize_by_w(self,img:np.ndarray):
        if img.ndim==2:
            h,w = img.shape
        else:
            h,w,_ = img.shape
        shrinking_ratio = self.default_w / float(w)
        img = cv2.resize(img, (self.default_w, int(h * shrinking_ratio)))
        return img

    def mask_image_gray(self,img:np.ndarray)->np.ndarray:

        ##cv2 (h,w,c)
        ori_h,ori_w= img.shape
        border_pixel = img[ori_h-1, ori_w - 1]
        ##numpy 填充
        mask = np.full((self.default_h, self.default_w),border_pixel)

        start_h = (self.default_h - ori_h)//2
        start_w = (self.default_w - ori_w)//2
        mask[start_h:start_h+ori_h,start_w:start_w+ori_w] = img
        return mask

    def mask_image_color(self,img:np.ndarray)->np.ndarray:
        ori_h, ori_w,ori_c = img.shape

        mask = np.zeros((self.default_h, self.default_w,3))

        start_h = (self.default_h - ori_h) // 2
        start_w = (self.default_w - ori_w) // 2
        mask[start_h:start_h + ori_h, start_w:start_w + ori_w,:] = img
        return mask


@PIPELINES.register_module
class RecognitionImageCV2Tensor(object):
    def __init__(self, img_channel=1):
        pass
        # assert  img_channel in [1,3]
        # self.img_channel = img_channel
        self.to_tensor_func = transforms.ToTensor()
    def __call__(self,data:{})->dict:
        img_array = data.get('img')
        data["img"] = self.to_tensor_func(img_array)
        return data
        # #cv2 (h,w,c)
        # if self.img_channel==3:
        #     #(h,w,c) -> (c,h,w)
        #     img_array = img_array.transpose((2,1,0))
        # elif self.img_channel==1:
        #     img_array = np.expand_dims(img_array,axis=0)
        # data["img"] = torch.from_numpy(img_array)
        # return data







