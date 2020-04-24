from texthub.utils import Config
from texthub.datasets import build_dataset

config_file = "./configs/testdatasetconfig.py"
cfg = Config.fromfile(config_file)
##
dataset = build_dataset(cfg.data.train)

# def main():
#     config_file = "../configs/testdatasetconfig.py"
#     cfg = Config.fromfile(config_file)
#     dataset = build_dataset(cfg.data.train)
