import bisect

from texthub.utils import print_log
from .registry import DATASETS
from torch.utils.data import ConcatDataset,Dataset
from .builder import build_dataset
import warnings
@DATASETS.register_module
class RatioBalancedDataset(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    def __init__(self, dataset_cfg_list: list, ratio_list: list):
        """
        对datasetlist里的dataset按照ratio_list里对应的比例组合，每个batch里的数据按按照比例采样的
        :param dataset_cfg_list: 数据集config列表
        :param ratio_list: 比例列表
        """
        assert sum(ratio_list) == 1 and len(dataset_cfg_list) == len(ratio_list)
        super(RatioBalancedDataset, self).__init__()

        self.dataset_list = [build_dataset(dataset_cfg) for dataset_cfg in dataset_cfg_list]

        self.ratio_list = ratio_list
        self.cumulative_sizes = self.cumsum(self.dataset_list)
        ##先实现1：1：1均等采样的方式

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.dataset_list[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes