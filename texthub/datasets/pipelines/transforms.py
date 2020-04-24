from ..registry import PIPELINES
import  numpy as np
import  math
from PIL import Image
import torch
import torchvision.transforms as transforms

@PIPELINES.register_module
class ResizeRecognitionImage(object):
    """
    文本识别的resize
    img_scale:(h,w)
    保持h不变的情况下,根据比例resizetup
    """
    def __init__(self,img_scale=None):
        assert  isinstance(img_scale,tuple) and len(img_scale)==2,"img_scale must be tuple(h,w)"
        self.default_h,self.default_w = img_scale

    def __call__(self, data:{}):
        # img
        img = data.get("img")

        w, h = img.size
        ratio = w / float(h)
        if math.ceil(self.default_h * ratio) > self.default_w:
            resized_w = self.default_w
        else:
            resized_w = math.ceil(self.default_h * ratio)
        resized_image = img.resize((resized_w, self.default_h), Image.BICUBIC)
        data["img"] = resized_image
        return data


@PIPELINES.register_module
class NormalizePADToTensor(object):
    """
    max_size:tuple(c,h,w)
    """
    def __init__(self, max_size, PAD_type='right'):
        assert  isinstance(max_size,tuple) and len(max_size)==3,"max_size must be tuple(c,h,w)"
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, data:{}):
        img = data.get('img')
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        data['img']= pad_img
        return data



# @PIPELINES.register_module
# class Normalize(object):
#     """Normalize the image.
#
#     Args:
#         mean (sequence): Mean values of 3 channels.
#         std (sequence): Std values of 3 channels.
#         to_rgb (bool): Whether to convert the image from BGR to RGB,
#             default is true.
#     """
#
#     def __init__(self, mean, std, to_rgb=True):
#         self.mean = np.array(mean, dtype=np.float32)
#         self.std = np.array(std, dtype=np.float32)
#         self.to_rgb = to_rgb
#
#     def __call__(self, results):
#         results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
#                                           self.to_rgb)
#         results['img_norm_cfg'] = dict(
#             mean=self.mean, std=self.std, to_rgb=self.to_rgb)
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += '(mean={}, std={}, to_rgb={})'.format(
#             self.mean, self.std, self.to_rgb)
#         return repr_str
