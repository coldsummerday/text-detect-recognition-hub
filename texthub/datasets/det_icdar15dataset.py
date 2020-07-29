import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from texthub.utils import print_log
from .registry import DATASETS
from .pipelines import Compose
import os
import cv2
@DATASETS.register_module
class IcdarDetectDataset(Dataset):
    def __init__(self,root:str,pipeline, img_channel=3,img_prefix = "imgs",gt_prefix="gts",line_flag=True):
        """
        if line_flag ==True,390,902,1856,902,1856,1225,390,1225,0,"金氏眼镜"
        Flase:237,48,237,75,322,75,322,48,明天
        """
        self.root = root
        self.img_channel = img_channel
        self.line_flag = line_flag

        self.img_path_fmt = os.path.join(root,img_prefix,"{}.jpg")
        self.gt_path_fmt = os.path.join(root,gt_prefix,"{}.txt")
        self.ids_list = self.load_index(root)
        self.pipeline = Compose(pipeline)


    def load_index(self,root_dir:str):
        """
        数据格式
        img/image_1.jpg
        img/image_2.jpg
        img/image_2.jpg
        ```

        gts/image_1.txt
        gts/image_2.txt
        """
        imgids = os.listdir(os.path.join(root_dir,'imgs'))
        imgids = [os.path.splitext(imgid)[0] for imgid in imgids]
        #check if gt exist
        exist_imgs = []
        for imgid in imgids:
            if os.path.exists(self.gt_path_fmt.format(imgid)):
                exist_imgs.append(imgid)
        return exist_imgs

    def _get_annotation(self, img_id: str) -> tuple:
        """
        icdar2017rctw fromat:390,902,1856,902,1856,1225,390,1225,0,"金氏眼镜"
        icdar2015 format:237,48,237,75,322,75,322,48,明天
        """
        boxes = []
        text_tags = []
        label_path = self.gt_path_fmt.format(img_id)
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.arcLength(box, True) > 0:
                        boxes.append(box)
                        text_tag,text_label = self._get_lable(params)

                        text_tags.append(text_tag)
                        # label = params[8]
                        # if label == '*' or label == '###':
                        #     text_tags.append(False)
                        # else:
                        #     text_tags.append(True)
                except Exception as e:
                    print_log('load label failed on {}'.format(label_path))
                    print_log(str(e))
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)
    def _get_lable(self,label_line_params:list):
        if self.line_flag:
            text_tag = label_line_params[8]
            text_label = label_line_params[9]
            #去除引号
            text_label = text_label[1:-1]
            if text_tag=='1':
                text_tag=False
            else:
                text_tag = True
        else:
            text_label = label_line_params[8]
            if text_label=="*" or text_label=='###':
                text_tag = False
            else:
                text_tag = True
        return text_tag,text_label



    def __getitem__(self, index):
        img_id = self.ids_list[index]
        img_path = self.img_path_fmt.format(img_id)

        text_polys, text_tags = self._get_annotation(img_id)
        img = cv2.imread(img_path, 1 if self.img_channel == 3 else 0)
        if self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img, score_map, training_mask = image_label(im, text_polys, text_tags, self.input_size,
        #                                                 self.shrink_ratio)

        data = {
            "img": img,
            "gt_polys": text_polys,
            "gt_tags":text_tags
        }
        return self.pipeline(data)


    def __len__(self):
        return len(self.ids_list)




#
# class Batch_Balanced_Dataset(object):
#     def __init__(self, dataset_list: list, ratio_list: list, module_args: dict,
#                  phase: str = 'train'):
#         """
#         对datasetlist里的dataset按照ratio_list里对应的比例组合，似的每个batch里的数据按按照比例采样的
#         :param dataset_list: 数据集列表
#         :param ratio_list: 比例列表
#         :param module_args: dataloader的配置
#         :param phase: 训练集还是验证集
#         """
#         assert sum(ratio_list) == 1 and len(dataset_list) == len(ratio_list)
#
#         self.dataset_len = 0
#         self.data_loader_list = []
#         self.dataloader_iter_list = []
#         all_batch_size = module_args['loader']['train_batch_size'] if phase == 'train' else module_args['loader'][
#             'val_batch_size']
#         for _dataset, batch_ratio_d in zip(dataset_list, ratio_list):
#             _batch_size = max(round(all_batch_size * float(batch_ratio_d)), 1)
#
#             _data_loader = DataLoader(dataset=_dataset,
#                                       batch_size=_batch_size,
#                                       shuffle=module_args['loader']['shuffle'],
#                                       num_workers=module_args['loader']['num_workers'])
#
#             self.data_loader_list.append(_data_loader)
#             self.dataloader_iter_list.append(iter(_data_loader))
#             self.dataset_len += len(_dataset)
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return min([len(x) for x in self.data_loader_list])
#
#     def __next__(self):
#         balanced_batch_images = []
#         balanced_batch_score_maps = []
#         balanced_batch_training_masks = []
#
#         for i, data_loader_iter in enumerate(self.dataloader_iter_list):
#             try:
#                 image, score_map, training_mask = next(data_loader_iter)
#                 balanced_batch_images.append(image)
#                 balanced_batch_score_maps.append(score_map)
#                 balanced_batch_training_masks.append(training_mask)
#             except StopIteration:
#                 self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
#                 image, score_map, training_mask = next(self.dataloader_iter_list[i])
#                 balanced_batch_images.append(image)
#                 balanced_batch_score_maps.append(score_map)
#                 balanced_batch_training_masks.append(training_mask)
#             except ValueError:
#                 pass
#
#         balanced_batch_images = torch.cat(balanced_batch_images, 0)
#         balanced_batch_score_maps = torch.cat(balanced_batch_score_maps, 0)
#         balanced_batch_training_masks = torch.cat(balanced_batch_training_masks, 0)
#         return balanced_batch_images, balanced_batch_score_maps, balanced_batch_training_masks
#


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts






