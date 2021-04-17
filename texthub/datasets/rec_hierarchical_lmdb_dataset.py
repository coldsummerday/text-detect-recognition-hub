import sys
import re
import lmdb
import numpy as np
import cv2
from torch.utils.data import Dataset
from texthub.utils import print_log

from .registry import DATASETS
from .pipelines import Compose
import random

@DATASETS.register_module
class HierarchicalLmdbDataset(Dataset):
    def __init__(self, root:[list], pipeline, charsets, default_w=100, default_h=32, rgb=False,
                 sensitive=True, batch_max_length=25):
        if type(root)==str:
            self.roots= [root]
        else:
            self.roots = root
        self.charsets = charsets
        self.rgb = rgb
        self.sensitive = sensitive

        self.batch_max_length = batch_max_length
        self.default_h = default_h
        self.default_w = default_w

        self.num_samples = 0
        # self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        # # set group flag for the sampler
        #
        # if not self.env:
        #     print_log('cannot create lmdb from %s' % (root), logger="root")
        #     sys.exit(0)
        self.pipeline = Compose(pipeline)
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset()

    def load_hierarchical_lmdb_dataset(self):
        lmdb_sets = {}
        data_set_idx = 0

        for dir_root in self.roots:

            env = lmdb.open(dir_root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            txn = env.begin(write=False)
            if not env:
                print_log('cannot create lmdb from %s' % (dir_root), logger="root")
                sys.exit(0)
            num_samples = int(txn.get('num-samples'.encode()))
            lmdb_sets[data_set_idx] = {"dir_root": dir_root, "env": env, \
                                      "txn": txn, "num_samples": num_samples}
            data_set_idx += 1
            self.num_samples += num_samples

        return lmdb_sets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index:int):
        assert index <= len(self), 'index range error'

        data_set_idx,sample_idx = self.find_set_index(index)
        img,label = self.get_lmdb_sample_info(self.lmdb_sets[data_set_idx]["txn"],sample_idx)
        if not self.sensitive:
            label = label.lower()
        # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
        out_of_char = f'[^{self.charsets}]'
        label = re.sub(out_of_char, '', label)
        if len(label) >= self.batch_max_length:
            label = label[:self.batch_max_length]
        data = {
            "img":img,
            "label":label
        }
        try:
            data = self.pipeline(data)
        except Exception as e:
            print_log("error index:{},error".format(index),logger="root")
            random_index = random.randint(0,self.num_samples)
            return self.__getitem__(random_index)

        return data


    def find_set_index(self,index:int):
        for i in range(len(self.roots)):
            if index<self.lmdb_sets[i]["num_samples"]:
                return i,index
            else:
                index = index-self.lmdb_sets[i]["num_samples"]
        return len(self.roots)-1,index







    def get_lmdb_sample_info(self,txn,index:int):
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

