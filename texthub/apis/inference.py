import torch

from PIL import Image
from ..utils import  Config
from ..modules import build_recognizer,build_detector
from ..core.utils.checkpoint import load_checkpoint
from ..datasets.pipelines import Compose
import numpy as np
import cv2

def init_recognizer(config,checkpoint=None,device=torch.device("cuda")):
    """Initialize a detector from config file.

        Args:
            config (str or :obj:`Config`): Config file path or the config
                object.
            checkpoint (str, optional): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
    if isinstance(config,str):
        config = Config.fromfile(config)
    elif not isinstance(config,Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_recognizer(
        config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint,map_location=device)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def init_detector(config,checkpoint=None,device=torch.device("cuda")):
    """Initialize a detector from config file.

        Args:
            config (str or :obj:`Config`): Config file path or the config
                object.
            checkpoint (str, optional): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
    if isinstance(config,str):
        config = Config.fromfile(config)
    elif not isinstance(config,Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(
        config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint,map_location=device)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def inference_detector(model,img:str):
    """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray ): Either image files or loaded
                images.

        Returns:
            If imgs is a str, a generator will be returned, otherwise return the
            detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.test_pipeline
    test_pipeline = Compose(test_pipeline)

    if isinstance(img,str):
        img = cv2.imread(img)
    elif isinstance(img,np.ndarray):
        img = img
    elif isinstance(img,Image):
        #TODO:将PIL改为CV2
        pass
    else:
        raise TypeError('img must be a PIL.Image or str or np.ndarray, '
                        'but got {}'.format(type(img)))

    ori_h,ori_w,ori_c = img.shape

    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    img_tensor = data['img'].unsqueeze(0).to(device)
    _,_,new_h,new_w = img_tensor.shape
    data_dict = dict(img=img_tensor)
    # forward the model
    with torch.no_grad():
        preds = model(data_dict,return_loss=False)
    pred_bbox_list,score_bbox_list = model.postprocess(preds)

    #pred_bbox_list(b,n,4,2)  [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] for bbox model
    batch_pred_bbox = pred_bbox_list[0]

    w_scale = float(ori_w) / new_w
    h_scale = float(ori_h) / new_h

    if type(batch_pred_bbox)==np.ndarray:
        ##bbox 情况，其4个点个数稳定
        batch_pred_bbox[:,:,0] *=w_scale
        batch_pred_bbox[:, :, 1] *= h_scale
    else:
        #polygon
        for polygon_array  in batch_pred_bbox:
            polygon_array[:, 0] = np.clip(
                np.round(polygon_array[:, 0] / new_w * ori_w), 0, ori_w)
            polygon_array[:, 1] = np.clip(
                np.round(polygon_array[:, 1] / new_h * ori_h), 0, ori_h)

    return batch_pred_bbox,score_bbox_list








def inference_recognizer(model,img:str):
    """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray ): Either image files or loaded
                images.

        Returns:
            If imgs is a str, a generator will be returned, otherwise return the
            detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.test_pipeline
    test_pipeline = Compose(test_pipeline)

    if isinstance(img,str):
        img = Image.open(img)
    elif isinstance(img,np.ndarray):
        ##原则上不需要装opencv库,但是如果传入的是opencv的对象,则需要进行转化
        import cv2
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif isinstance(img,Image):
        img = img
    else:
        raise TypeError('img must be a PIL.Image or str or np.ndarray, '
                        'but got {}'.format(type(img)))
    #rgb2gray
    img = img.convert("L")

    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    img_tensor = data['img'].unsqueeze(0).to(device)
    data["img"] = img_tensor
    # forward the model
    with torch.no_grad():
        preds = model(data,return_loss=False)
    preds = model.postprocess(preds)
    return preds[0]






