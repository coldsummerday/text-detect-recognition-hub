import os.path as osp
import os
import sys

this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.modules import build_recognizer
from texthub.utils import  Config
from texthub.datasets import build_dataset




config_file = "./configs/receipt/recognition/scatter/scatter_hr_hierachicaldataset.py"
config = Config.fromfile(config_file)
model = build_recognizer(config.model)
dataset = build_dataset(config.data.train)
import torch
b=torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            pin_memory=True,
        )
data = b.__iter__().__next__()
a=model(data,return_loss=True)
