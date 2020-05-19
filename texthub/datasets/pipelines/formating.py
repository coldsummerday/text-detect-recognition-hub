from ..registry import PIPELINES


@PIPELINES.register_module
class Collect(object):
    """
    Collect data from the loader relevant to the specific task.
    """
    def __init__(self,
                 keys,
                 meta_keys=( 'ori_shape', 'img_shape',
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        # img_meta = {}
        # for key in self.meta_keys:
        #     if key not in results.keys():
        #         results[key]=None
        #         continue
        #     img_meta[key] = results[key]
        # data["img_meta"] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)
