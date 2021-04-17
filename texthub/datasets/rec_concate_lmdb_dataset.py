
import cv2
from torch.utils.data import Dataset,ConcatDataset
import copy
from .registry import DATASETS
from .builder import build_dataset


import re
import six
from PIL import Image
import lmdb
import numpy as np
from texthub.utils import print_log
from .pipelines import Compose
import random

@DATASETS.register_module
class ConcateLmdbDataset(ConcatDataset):
    def __init__(self,**kwargs):
        base_dataset = kwargs.pop("base_dataset_type")
        roots = kwargs.pop("root")
        kwargs.setdefault('type',base_dataset)
        datasets = []
        for root in roots:
            base_kwargs = copy.deepcopy(kwargs)
            base_kwargs.setdefault("root",root)
            datasets.append(build_dataset(base_kwargs))
        super(ConcateLmdbDataset, self).__init__(datasets=datasets)




@DATASETS.register_module
class LmdbWorkersDataset(Dataset):
    def __init__(self,root,pipeline,charsets,
                 default_w = 100,default_h=32,
                 data_filtering_off=True,
                 rgb =False,
                 sensitive=True,
                 batch_max_length=25):
        super(LmdbWorkersDataset, self).__init__()
        self.root = root
        self.charsets = charsets
        self.rgb = rgb
        self.sensitive = sensitive

        self.batch_max_length = batch_max_length
        self.default_h = default_h
        self.default_w = default_w


        with lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            if data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.charsets}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)
        self.pipeline = Compose(pipeline)

    #     self._set_group_flag()
    # def _set_group_flag(self):
    #     """Set flag according to image aspect ratio.
    #
    #     Images with aspect ratio greater than 1 will be set as group 1,
    #     otherwise group 0.
    #     """
    #     self.flag = np.zeros(len(self), dtype=np.uint8)
    #     for i in range(len(self)):
    #         # img_info = self.img_infos[i]
    #         # if img_info['width'] / img_info['height'] > 1:
    #         self.flag[i] = 1

    def __len__(self):
        return self.nSamples

    def open_lmdb(self):
        self.env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]
        if not hasattr(self, 'env'):
            self.open_lmdb()
        img, label = self.get_lmdb_sample_info(self.env, index)
        if not self.sensitive:
            label = label.lower()
        # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
        out_of_char = f'[^{self.charsets}]'
        label = re.sub(out_of_char, '', label)
        if len(label) >= self.batch_max_length:
            label = label[:self.batch_max_length]
        data = {
            "img": img,
            "label": label
        }
        try:
            data = self.pipeline(data)
        except Exception as e:
            # print_log("error index:{}".format(index), logger="root")
            random_index = random.randint(0, self.nSamples)
            return self.__getitem__(random_index)

        return data

    def get_lmdb_sample_info(self,env,index:int):
        with env.begin(write=False) as txn:
            if index==0:
                index+=1
            label_key = 'label-%09d'.encode() % (index)
            label = txn.get(label_key)
            if label is None:
                print_log('error to find the index:{}'.format(index), logger="root")
                return self.get_none_image()
            label = label.decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            img = self.get_img_data(imgbuf)
        return img, label

    def get_none_image(self):
        if self.rgb:
            img = np.zeros((self.default_h, self.default_w, 3))
        else:
            img = np.zeros((self.default_h, self.default_w, 1))
        return  img,""

    def get_img_data(self,imgbuf):
        try:
            imgdata = np.frombuffer(imgbuf, dtype='uint8')
            if self.rgb:
                img = cv2.imdecode(imgdata, cv2.IMREAD_COLOR)
            else:
                img = cv2.imdecode(imgdata, cv2.IMREAD_GRAYSCALE)
        except IOError:

            # make dummy image and dummy label for corrupted image.
            if self.rgb:
                img = np.zeros((self.default_h, self.default_w, 3))
            else:
                img = np.zeros((self.default_h, self.default_w, 1))
            # label = '[dummy_label]'
        return img

@DATASETS.register_module
class LmdbPILWorkersDataset(Dataset):
    def __init__(self,root,pipeline,charsets,
                 default_w = 100,default_h=32,
                 data_filtering_off=True,
                 rgb =False,
                 sensitive=True,
                 batch_max_length=25):
        super(LmdbPILWorkersDataset, self).__init__()
        self.root = root
        self.charsets = charsets
        self.rgb = rgb
        self.sensitive = sensitive

        self.batch_max_length = batch_max_length
        self.default_h = default_h
        self.default_w = default_w

        with lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False).begin(
                write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            if data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.charsets}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)
        self.pipeline = Compose(pipeline)

    #     self._set_group_flag()
    # def _set_group_flag(self):
    #     """Set flag according to image aspect ratio.
    #
    #     Images with aspect ratio greater than 1 will be set as group 1,
    #     otherwise group 0.
    #     """
    #     self.flag = np.zeros(len(self), dtype=np.uint8)
    #     for i in range(len(self)):
    #         # img_info = self.img_infos[i]
    #         # if img_info['width'] / img_info['height'] > 1:
    #         self.flag[i] = 1

    def __len__(self):
        return self.nSamples

    def open_lmdb(self):
        self.env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]
        if not hasattr(self, 'env'):
            self.open_lmdb()
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.rgb:
                    img = Image.new('RGB', (self.default_h, self.default_w))
                else:
                    img = Image.new('L', (self.default_h, self.default_w))
                label = '[dummy_label]'

            if not self.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.charsets}]'
            label = re.sub(out_of_char, '', label)
        ##在dataloader前变label 为tensor,保证在data_parallel时能正确地被均分
        data = {
            "img":img,
            "label":label
        }
        try:
            data = self.pipeline(data)
        except Exception as e:
            # print_log("error index:{}".format(index), logger="root")
            random_index = random.randint(0, self.nSamples)
            return self.__getitem__(random_index)

        return data









