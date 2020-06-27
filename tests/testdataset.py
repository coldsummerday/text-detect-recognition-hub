from texthub.utils import Config
from texthub.datasets import build_dataset

config_file = "./configs/fourstagerecogition/tps_vgg_lstm_attention.py"
cfg = Config.fromfile(config_file)
##
dataset = build_dataset(cfg.data.train)
import torch
b=torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            shuffle=True,
            pin_memory=True
        )
b.__iter__().__next__()
# def main():
#     config_file = "../configs/testdatasetconfig.py"
#     cfg = Config.fromfile(config_file)
#     dataset = build_dataset(cfg.data.train)
