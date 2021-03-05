import numpy as np
# from sortedcontainers import SortedSet
import string
import torch
from ..registry import PIPELINES


@PIPELINES.register_module
class CTCChineseCharsetConverter(object):
    def __init__(self,charsets,batch_max_length=32):
        corups = charsets
        self.blank = 0
        self.unknown =1
        self.blank_char ='\t'
        self.unknown_char = '\n'
        self.case_sensitive = False

        self._corpus = self._filter_corpus(corups)
        # self._corpus = SortedSet(self._filter_corpus(corups))
        if self.blank_char in self._corpus:
            self._corpus.remove(self.blank_char)
        if self.unknown_char in self._corpus:
            self._corpus.remove(self.unknown_char)
        self._charset = list(self._corpus)
        self._charset.insert(self.blank, self.blank_char)
        self._charset.insert(self.unknown, self.unknown_char)
        self._charset_lut = {char: index
                             for index, char in enumerate(self._charset)}
        self.max_size = batch_max_length
    def _filter_corpus(self, iterable):
        corups = []
        for char in iterable:
            if not self.case_sensitive:
                char = char.upper()
            corups.append(char)
        return corups

    def __call__(self, data: {}):
        label = data.get('label')
        # device = img.device
        length = len(label)
        length_tensor = np.array(length, dtype=np.int32)
        label_tensor = torch.from_numpy(self.string_to_label(label))
        data["ori_label"] = label
        data['label'] = label_tensor
        data["length"] =length_tensor
        return data

    def index(self, x):
        target = x
        if not self.case_sensitive:
            target = target.upper()
        return self._charset_lut.get(target, self.unknown)


    def __len__(self):
        return len(self._charset)

    def string_to_label(self, string_input):
        length = max(self.max_size, len(string_input))
        target = np.zeros((length, ), dtype=np.int32)
        for index, c in enumerate(string_input[:length]):
            value = self.index(c)
            target[index] = value
        return target

    def label_to_string(self, label):
        ingnore = [self.unknown, self.blank]
        return "".join([self._charset[i] for i in label if i not in ingnore])