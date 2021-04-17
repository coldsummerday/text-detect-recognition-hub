import  inspect
from functools import partial

"""
通过注册器的方式,将各个组件注册到注册表中,方便从配置文件中动态读取加载类
"""
class Registry(object):
    def __init__(self,name):
        self._name = name
        self.__module__dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self.__module__dict.keys()))
        return format_str

    @property
    def module_dict(self):
        return self.__module__dict

    @property
    def name(self):
        return self._name

    def get(self,key):
        return self.__module__dict.get(key)

    def _register_module(self,module_class,force=False):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class,but got {}".format(type(module_class)))
        module_name = module_class.__name__
        if module_name in self.__module__dict.keys():
            raise KeyError("{} is already registered in {}".format(
                module_name,self.name
            ))
        self.__module__dict[module_name] = module_class

    def register_module(self,cls=None,force=None):
        if cls==None:
            #partial 偏函数,它返回一个偏函数对象，
            # 这个对象和 func 一样，可以被调用，
            # 同时在调用的时候可以指定位置参数 (args) 和 关键字参数(*kwargs)。
            # 如果有更多的位置参数提供调用，它们会被附加到 args 中。
            # 如果有额外的关键字参数提供，它们将会扩展并覆盖原有的关键字参数。它的实现大致等同于如下代码
            return partial(self.register_module, force=force)
        self._register_module(cls,force=force)
        return cls

def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type,str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)


