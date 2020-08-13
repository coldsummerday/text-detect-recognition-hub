from  torch.utils.data.dataloader import DataLoader
class DataPrefetcher(object):
    """
    包装dataloader 提前预读文件
    """
    def __init__(self, loader:DataLoader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()
        self.sampler = loader.sampler

    ##BUG：
    ##这样可能导致数据在最后一个batch的时候一直都是最后的数据
    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)
