import  numpy as np
import  torch
from tests.resnet_aster import ResNet_ASTER
from texthub.modules.label_heads.asterheads import AsterAttentionRecognitionHead
from texthub.datasets import CharsetDict

ChineseCharset = CharsetDict["ChineseCharset"]


x = torch.randn(3, 3, 32, 100)
net = ResNet_ASTER(with_lstm=True)
encoder_feat = net(x)
print(encoder_feat.size())

head = AttentionRecognitionHead(512,256,256,ChineseCharset)
label_tensor = torch.ones((3,25))

input_dict = {
    "img":encoder_feat,
    "label":label_tensor
}

x = head(input_dict,return_loss = True)
rec_preds = head.beam_search(input_dict)
print(rec_preds.shape)
print(x.shape)