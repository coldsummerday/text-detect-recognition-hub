from ..registry import PIPELINES
import  numpy as np
import  math
from PIL import Image
import torch
import torchvision.transforms as transforms
# ABCs from collections will be deprecated in python 3.8+,
# while collections.abc is not available in python 2.7
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc
import cv2
import numbers
import pyclipper

import Polygon as plg
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


@PIPELINES.register_module
class Ndarray2tensor(object):
    def __init__(self):
        self.transforms = transforms.ToTensor()
        pass
    def __call__(self,data:dict):
        assert type(data.get("img"))==np.ndarray
        data['img'] = self.transforms(Image.fromarray(data['img']))
        return data

@PIPELINES.register_module
class Gt2SameDim(object):
    def __init__(self,max_label_num=150):
        self.max_label_num = max_label_num

    def __call__(self,data:dict):
        assert type(data.get("gt_polys"))==np.ndarray
        gt=np.zeros((self.max_label_num,4,2))
        gt_polys_array = data.get("gt_polys")
        length = len(gt_polys_array)
        gt[:length] = gt_polys_array
        data['gt_polys'] = gt
        return data




@PIPELINES.register_module
class GenerateTrainMask(object):
    """
    shrink_ratio: gt收缩的比例
    """
    def __init__(self,shrink_ratio=0.5):
        self.shrink_ratio = shrink_ratio


    def __call__(self,data:dict):
        h, w, c = data["img"].shape
        text_polys = data["gt_polys"]
        text_tags = data["gt_tags"]
        training_mask = np.ones((h, w), dtype=np.uint8)
        score_maps = []
        for i in (1, self.shrink_ratio):
            score_map, training_mask = self.generate_rbox((h, w), text_polys, text_tags, training_mask, i)
            score_maps.append(score_map)
        score_maps = np.array(score_maps, dtype=np.float32)
        data["labels"] = score_maps
        data["training_mask"] = training_mask
        return data



    def generate_rbox(self,im_size,text_polys, text_tags, training_mask, shrink_ratio):
        """
        生成mask图，白色部分是文本，黑色是背景
        :param im_size: 图像的h,w
        :param text_polys: 框的坐标
        :param text_tags: 标注文本框是否参与训练
        :param training_mask: 忽略标注为 DO NOT CARE 的矩阵
        :return: 生成的mask图
        """
        h, w = im_size
        score_map = np.zeros((h, w), dtype=np.uint8)
        for i, (poly, tag) in enumerate(zip(text_polys, text_tags)):
            try:
                poly = poly.astype(np.int)
                # d_i = cv2.contourArea(poly) * (1 - shrink_ratio * shrink_ratio) / cv2.arcLength(poly, True)
                d_i = cv2.contourArea(poly) * (1 - shrink_ratio) / cv2.arcLength(poly, True) + 0.5
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked_poly = np.array(pco.Execute(-d_i))
                cv2.fillPoly(score_map, shrinked_poly, i + 1)
                if not tag:
                    cv2.fillPoly(training_mask, shrinked_poly, 0)
            except:
                print(poly)
        return score_map, training_mask




@PIPELINES.register_module
class RandomRotate(object):
    """
    从给定的角度中选择一个角度，对图片和文本框进行旋转

    :param degrees: 角度，可以是一个数值或者list
    :param same_size: 是否保持和原图一样大

    """
    def __init__(self,degrees,same_size =True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.degrees= degrees
        self.same_size = same_size
    def __call__(self,data:dict)->dict:
        img = data.get('img')
        text_polys = data.get("gt_polys")
        w = img.shape[1]
        h = img.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        data["img"] = rot_img
        data["gt_polys"] = np.array(rot_text_polys, dtype=np.float32)
        return data

@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self,data:dict):
        flip = True if np.random.rand() < self.flip_ratio else False
        if flip:
            #需要翻转
            if self.direction =="horizontal":
                img = data["img"]
                text_polys = data.get("gt_polys")
                flip_text_polys = text_polys.copy()
                flip_im = cv2.flip(img, 1)
                h, w, _ = flip_im.shape
                flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]
                data["img"] = flip_im
                data["gt_polys"] = flip_text_polys
            if self.direction =="vertical":
                img = data["img"]
                text_polys = data.get("gt_polys")
                flip_text_polys = text_polys.copy()
                flip_im = cv2.flip(img, 0)
                h, w, _ = flip_im.shape
                flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
                data["img"] = flip_im
                data["gt_polys"] = flip_text_polys
        return data







@PIPELINES.register_module
class DetectResize(object):
    """
    resize the img and bbox or poly
    img_scale should be the tuple (600,600)
    """

    def __init__(self,img_scale,keep_ratio=False):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio



    def __call__(self,data:dict):
        assert  isinstance(self.img_scale,tuple) and len(self.img_scale)==2
        img = data.get('img')

        if self.keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = img.shape
            max_h = max(h, self.img_scale[0])
            max_w = max(w, self.img_scale[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = img.copy()
            img = im_padded
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, self.img_scale)
        data['img'] = img

        if "gt_polys" in data.keys():
            ##gt_polys (n,4,2),4个点(x,y)
            text_polys = data.get("gt_polys")
            text_polys = text_polys.astype(np.float32)
            w_scale = self.img_scale[0] / float(ori_w)
            h_scale = self.img_scale[1] / float(ori_h)
            text_polys[:, :, 0] *= w_scale
            text_polys[:, :, 1] *= h_scale
            data['gt_polys'] = text_polys
        if "gt_bbox" in data.keys():
            gt_bbox = data.get("gt_bbox")
            ##TODO:gt_bbox resize
            pass
        return data



    def resize_img(self,img):
        if self.keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = img.shape
            max_h = max(h, self.img_scale[0])
            max_w = max(w, self.img_scale[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = img.copy()
            img = im_padded



    def resize(self, im: np.ndarray, text_polys: np.ndarray,
               input_size: numbers.Number or list or tuple or np.ndarray, keep_ratio: bool = False) -> tuple:
        """
        对图片和文本框进行resize
        :param im: 图片
        :param text_polys: 文本框
        :param input_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param keep_ratio: 是否保持长宽比
        :return: resize后的图片和文本框
        """
        if isinstance(input_size, numbers.Number):
            if input_size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            input_size = (input_size, input_size)
        elif isinstance(input_size, list) or isinstance(input_size, tuple) or isinstance(input_size, np.ndarray):
            if len(input_size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            input_size = (input_size[0], input_size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        if keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, input_size[0])
            max_w = max(w, input_size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, input_size)
        w_scale = input_size[0] / float(w)
        h_scale = input_size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale
        return im, text_polys



@PIPELINES.register_module
class CheckPolys(object):
    """
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    """
    def __init__(self):
        pass

    def __call__(self,data):
        """
        :param data: dict contains (img:np.ndarray,gt_polys:np.ndarray)
        :return:
        """
        h,w,_ = data['img'].shape
        polys = data['gt_polys']
        if polys.shape[0] == 0:
            return polys
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

        validated_polys = []
        for poly in polys:
            p_area = cv2.contourArea(poly)
            if abs(p_area) < 1:
                continue
            validated_polys.append(poly)
        data['gt_polys']=np.array(validated_polys)
        return data
    def __repr__(self):
        return self.__class__.__name__



def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = collections_abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True





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
