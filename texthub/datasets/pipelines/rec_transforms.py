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

import  random
from PIL import Image,ImageEnhance,ImageFont,ImageDraw
from copy import deepcopy
import  numpy as np


import os
this_path = os.path.split(os.path.realpath(__file__))[0]
from .text_image_aug.augment import tia_distort, tia_stretch, tia_perspective


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
        mask = np.full((self.default_h, self.default_w),border_pixel).astype("float32")

        start_h = (self.default_h - ori_h)//2
        start_w = (self.default_w - ori_w)//2
        mask[start_h:start_h+ori_h,start_w:start_w+ori_w] = img
        return mask

    def mask_image_color(self,img:np.ndarray)->np.ndarray:
        ori_h, ori_w,ori_c = img.shape

        mask = np.zeros((self.default_h, self.default_w,3)).astype("float32")

        start_h = (self.default_h - ori_h) // 2
        start_w = (self.default_w - ori_w) // 2
        mask[start_h:start_h + ori_h, start_w:start_w + ori_w,:] = img
        return mask


class GenRecognitionImageCV2(object):
    """
    文本识别的resize
    img_scale:(h,w)
    尽量不resize 图片，只将原始图片贴到mask上，保持图像原来清晰度
    """
    def __init__(self,img_scale=None,img_channel = 1,font_file="sont.ttf",font_color="black"):
        assert  isinstance(img_scale,tuple) and len(img_scale)==2,"img_scale must be tuple(h,w)"
        self.default_h,self.default_w = img_scale
        self.img_channel = img_channel
        self.font_handler = ImageFont.truetype(os.path.join(this_path,'../resources/',font_file),int(self.default_h/4*3),encoding="utf-8")
        # self.font_handler = ImageFont.truetype("/home/zhou/project/receipt/text-detect-recognition-hub/texthub/datasets/resources/sont.ttf",int(self.default_h/4*3),encoding="utf-8")
        self.font_color_value = 0
        if font_color=="white":
            self.font_color_value = 255


    def __call__(self, data:{}):

        # img
        img = data["img"]
        text = data["text"]
        if img.ndim==2:
            ori_h,ori_w = img.shape
            border_pixel = img[ori_h - 1, ori_w - 1]
            mask = np.full((self.default_h, self.default_w), border_pixel).astype("float32")
            pilimg = Image.fromarray(mask)
            draw = ImageDraw.Draw(pilimg)
            draw.text((0, 0), text, self.font_color_value, font=self.font_handler)
            gen_img = np.array(pilimg)
            data["gen_img"]=gen_img
            return data
        else:
            ori_h,ori_w,_ = img.shape
            # cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = Image.new("RGBA",(self.default_w,self.default_h),(int(img[ori_h - 1, ori_w - 1][2]),int(img[ori_h - 1, ori_w - 1][1]),int(img[ori_h - 1, ori_w - 1][0]),0))
            # border_pixel = img[ori_h - 1, ori_w - 1]
            # mask = np.zeros((self.default_h, self.default_w,3)).astype("float32")
            # mask[:,:,0]= np.full((self.default_h, self.default_w), border_pixel[0]).astype("float32")
            # mask[:, :, 1] = np.full((self.default_h, self.default_w), border_pixel[1]).astype("float32")
            # mask[:, :, 2] = np.full((self.default_h, self.default_w), border_pixel[2]).astype("float32")
            # cv2img = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            #
            # pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(mask)  # 图片上打印
            draw.text((0, 0), text, (self.font_color_value, self.font_color_value, self.font_color_value), font=self.font_handler)
            gen_img = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
            data["gen_img"] = gen_img
            return data
        return data

@PIPELINES.register_module
class GennerateBlurImage(object):
    def __init__(self):
        self.funcs = {
      "contrast": lambda img: self.contrast(img),
      "gaussian_blur": lambda img: self.gaussian_blur(img),
      "down_up_sample": lambda img: self.down_up_sample(img),
    }


    def __call__(self,data:dict):
        img = data["img"]
        hr_img = deepcopy(img)
        augmentations = ['contrast', 'gaussian_blur', 'down_up_sample']
        random.shuffle(augmentations)
        for item_aug in augmentations:
            if random.random() > 0.5:
                img = self.funcs[item_aug](img)
        data["lr_img"] = img.astype("float32")
        data["img"] = hr_img.astype("float32")
        return data



    def gaussian_blur(self, img:np.ndarray)->np.ndarray:
        g_kernel = random.randint(1, 5) * 2 + 1
        img = cv2.GaussianBlur(img, ksize=(g_kernel, g_kernel), sigmaX=0, sigmaY=0)
        return img


    def contrast(self, img:np.ndarray)->np.ndarray:
        copy_img=Image.fromarray(np.uint8(img))
        enhance_img= ImageEnhance.Contrast(copy_img).enhance(1 + 5 * random.choice([-1, 1]))
        return np.asarray(enhance_img)

    def down_up_sample(self, img:np.ndarray):
        if len(img.shape)==3:
            ori_h, ori_w,ori_c = img.shape
        else:
            ori_h, ori_w = img.shape
        size = (ori_w, ori_h)
        new_size = (int(ori_w / (random.random() * 2 + 1)), int(ori_h/ (random.random() * 2 + 1)))
        ##down
        img = cv2.resize(img,new_size)

        #up_sample
        img = cv2.resize(img,size)
        return img


@PIPELINES.register_module
class RecognitionImageCV2Tensor(object):
    def __init__(self, img_channel=1):
        # assert  img_channel in [1,3]
        # self.img_channel = img_channel
        self.to_tensor_func = transforms.ToTensor()
    def __call__(self,data:dict)->dict:
        for key,value in data.items():

            if key.find("img")!=-1:
                new_value = self.to_tensor_func(np.array(value))
                data[key] = new_value
        return data
        # img_array = data.get('img')
        # data["img"] = self.to_tensor_func(img_array)
        # return data
        # #cv2 (h,w,c)
        # if self.img_channel==3:
        #     #(h,w,c) -> (c,h,w)
        #     img_array = img_array.transpose((2,1,0))
        # elif self.img_channel==1:
        #     img_array = np.expand_dims(img_array,axis=0)
        # data["img"] = torch.from_numpy(img_array)
        # return data

@PIPELINES.register_module
class RecognitionNormalizeTensor(object):
    def __init__(self,RGB_MEAN=[122.67891434, 116.66876762, 104.00698793]):
        if RGB_MEAN[0]>1 and RGB_MEAN[0]<256:
            RGB_MEAN = [value/256 for value in RGB_MEAN]

        self.normal_func = transforms.Normalize(mean=RGB_MEAN,std=[0.229, 0.224, 0.225])
    def __call__(self,data:dict)->dict:
        for key,value in data.items():
            if key.find("img")!=-1:
                new_value = self.normal_func(value)
                data[key] = new_value
        return data



##TODO:BUG 并没有弯曲文本的作用
@PIPELINES.register_module
class TIATransform(object):
    ##概率必须是1
    def __init__(self,prob=1):
        self.prob = prob
    
    def tiaFunc(self,img:np.ndarray)->np.ndarray:

        img_height, img_width = img.shape[0:2]
        new_img = img

        if random.random() <= self.prob and img_height >= 20 and img_width >= 20:
            try:
                new_img = tia_distort(new_img, random.randint(3, 6))
            except:
                pass
                # print(
                #     "Exception occured during tia_distort, pass it...")
        if random.random() <= self.prob and img_height >= 20 and img_width >= 20:
            try:
                new_img = tia_stretch(new_img, random.randint(3, 6))
            except:
                pass
                # print(
                #     "Exception occured during tia_stretch, pass it...")
        if random.random() <= self.prob:
            try:
                new_img = tia_perspective(new_img)
            except:
                pass
                # print(
                #     "Exception occured during tia_perspective, pass it...")
        return new_img
    def __call__(self,data:dict)->dict:
        for key,value in data.items():
            if key.find("img")!=-1:
                
                new_value = self.tiaFunc(value)
                data[key] = new_value
        return data

@PIPELINES.register_module
class CV2ImageToGray(object):
    def __init__(self):
        pass
    def __call__(self,data:dict)->dict:
        for key,value in data.items():
            if key.find("img")!=-1 and len(value.shape)==3 and value.shape[2]==3:

                new_value =cv2.cvtColor(value,cv2.COLOR_RGB2GRAY)
                data[key] = new_value
        return data

    

            
    



@PIPELINES.register_module
class GasussNoise(object):
    def __init__(self,prob=0.4,mean=0,var =0.1):
        self.prob = prob
        self.mean = mean
        self.var = var
    def noiseFunc(self,img:np.ndarray)->np.ndarray:
        """
        Gasuss noise
        """

        noise = np.random.normal(self.mean, self.var**0.5, img.shape)
        out = img + 0.5 * noise
        out = np.clip(out, 0, 255)
        out = np.uint8(out)
        return out

    def __call__(self,data:dict)->dict:
        if random.random() > self.prob:
            return data

        for key,value in data.items():
            if key.find("img")!=-1:
                new_value = self.noiseFunc(value)
                data[key] = new_value
        return data

@PIPELINES.register_module
class Jitter(object):
    def __init__(self,prob=0.4):
        self.prob = prob

    def jitterFunc(self,img:np.ndarray):
        w, h, _ = img.shape
        if h > 10 and w > 10:
            thres = min(w, h)
            s = int(random.random() * thres * 0.01)
            src_img = img.copy()
            for i in range(s):
                img[i:, i:, :] = src_img[:w - i, :h - i, :]
            return img
        else:
            return img
    def __call__(self,data:dict)->dict:
        if random.random() > self.prob:
            return data

        for key,value in data.items():
            if key.find("img")!=-1:
                new_value = self.jitterFunc(value)
                data[key] = new_value
        return data



@PIPELINES.register_module
class GaussianBlur(object):
    def __init__(self,prob=0.4):
        self.prob = prob

    def blurFunc(self,img:np.ndarray)->np.ndarray:
        h, w, _ = img.shape
        if h > 10 and w > 10 :
            return cv2.GaussianBlur(img, (5, 5), 1)
        else:
            return img
    def __call__(self,data:dict)->dict:
        if random.random() > self.prob:
            return data

        for key,value in data.items():
            if key.find("img")!=-1:
                new_value = self.blurFunc(value)
                data[key] = new_value
        return data








