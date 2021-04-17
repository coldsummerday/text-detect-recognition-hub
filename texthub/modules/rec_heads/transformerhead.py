import torch
import torch.nn as nn
import math

from ..registry import HEADS
from ..backbones.rec_encoders.transformer import TransformerDecoder
from ..utils.moduleinit import kaiming_init,constant_init,xavier_init
import torch.nn.functional as F
from ..labelconverter import AttnLabelConverter

@HEADS.register_module
class TransfomerHead(nn.Module):
    def __init__(self,decoder_dict:dict,
                     max_len_labels:int,
                    hidden_dim:int,
                    charsets:str,
                    ignore_index:int=0):
        super(TransfomerHead, self).__init__()

        self.converter = AttnLabelConverter(charsets,max_len_labels=max_len_labels,ignore_id=ignore_index)
        """
        #在attentionlabelconvert 中,
        list_token = ['[GO]', '[s]']  
        list_character = list(charsets)
        self.character = list_token + list_character
        所以num_classes 应该为charset+2  "go,s"
        """
        num_classes = len(self.converter.character)

        self.decoder = TransformerDecoder(**decoder_dict)
        self.max_len_labels = max_len_labels
        self.num_steps = max_len_labels+1

        self.generator_layer = nn.Linear(in_features=hidden_dim,out_features=num_classes)
        self.embedding_layer = nn.Embedding(num_embeddings=num_classes+1,embedding_dim=hidden_dim,padding_idx=ignore_index)

        self.ignore_index= ignore_index
        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)


    
    def init_weights(self,pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    xavier_init(m)
                elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
                    kaiming_init(m, is_rnn=True)

    def forward(self,data:dict,return_loss:bool,**kwargs):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)


    def forward_train(self,data:dict):
        src = data['img']
        texts = data["label"]

        target = texts[:, 1:]  # without [GO] Symbol
        text = texts[:, :-1]  # align with Attention.forward

        src_mask = None
        tgt = self.text_embedding(text)
        tgt_mask = (self.pad_mask(text) | self.order_mask(text))

        out = self.decoder(tgt, src, tgt_mask, src_mask)
        pred = self.generator_layer(out)

        loss_recog = self.loss_func(pred.view(-1, pred.shape[-1]), target.contiguous().view(-1))
        return dict(loss_reco=loss_recog, loss=loss_recog)


    def forward_test(self,data:dict):
        src_tensor = data["img"]
        batch_size = src_tensor.size(0)
        ##预测用空字符串
        texts = torch.LongTensor(batch_size, 1).fill_(self.ignore_index).to(src_tensor.device)
        src_mask = None
        out = None
        for _ in range(self.num_steps):
            tgt = self.text_embedding(texts)
            tgt_mask = self.order_mask(texts)
            out = self.decoder(tgt, src_tensor, tgt_mask, src_mask)

            out = self.generator_layer(out)
            next_text = torch.argmax(out[:, -1:, :], dim=-1)
            texts = torch.cat([texts, next_text], dim=-1)
        preds = out
        return preds

    def postprocess(self,preds:torch.Tensor)->([],[]):
        batch_size = preds.size(0)
        probs = F.softmax(preds, dim=2)
        max_probs, indexes = probs.max(dim=2)
        preds_str = []
        preds_prob = []
        for i, pstr in enumerate(self.converter.decode(indexes,[self.max_len_labels]*batch_size)):
            str_len = len(pstr)
            if str_len == 0:
                prob = 0
            else:
                prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
            preds_prob.append(prob)
            preds_str.append(pstr)
        return preds_str, preds_prob


    def pad_mask(self,text:torch.Tensor):
        pad_mask = (text==self.ignore_index)
        pad_mask[:,0]=False
        pad_mask = pad_mask.unsqueeze(1)
        return  pad_mask
    def order_mask(self,text:torch.Tensor):
        t = text.size(1)
        order_mask = torch.triu(torch.ones(t,t),diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0).to(text.device)
        return order_mask

    def text_embedding(self,text:torch.Tensor):
        tgt = self.embedding_layer(text)
        tgt *= math.sqrt(tgt.size(2))
        return tgt














