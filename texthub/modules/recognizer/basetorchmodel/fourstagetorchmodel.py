import torch.nn as nn
from .base import BaseTorchRecognizer
from ...builder import build_backbone, build_img_trans, build_converter, build_prediction, build_sequence


class FourStageModel(BaseTorchRecognizer):
    """
    four stage recognition model with:trans->feature_extraction->sequence_modeling->prediction
    base on the paper:What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis.
    https://arxiv.org/pdf/1904.01906.pdf
    """

    def __init__(
            self,
            backbone,
            sequence=None,
            prediction=None,
            labelconverter=None,
            transformation=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None):
        super(FourStageModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.prediction = build_prediction(prediction)
        self.sequenceModeling = build_sequence(sequence)

        if transformation != None:
            self.img_transformation = build_img_trans(transformation)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained)
        ##TODO:未完待续
