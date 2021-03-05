import  torch
import torch.nn as nn
import torch.nn.functional as F
from ..sequencerecognition.bilstm import BidirectionalLSTM
from .labelconverter import AttnLabelConverter
from  ..utils.moduleinit import normal_init
from ..registry import HEADS

@HEADS.register_module
class ScatterHead(nn.Module):
    """
    ctc refine + 5 ScBlocks
    """
    def __init__(self,input_size:int,hidden_size:int,charsets:str,
                 sc_nums:int=5,max_len_labels:int=50,
                 ctc_loss_ratio:float=0.1,attn_loss_ratio:float=1):
        super(ScatterHead, self).__init__()
        self.converter = AttnLabelConverter(charsets)
        num_class = len(self.converter.character)
        ##ctc_refine
        self.ctc_refine_fc = nn.Linear(input_size,input_size)
        self.ctc_decoder = nn.Linear(input_size,num_class)

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.scblocks = nn.Sequential()
        for i in range(sc_nums):
            self.scblocks.add_module("sc_module_{}".format(i),SelectiveContextualBlock(
                input_size,hidden_size,num_class
            ))

        self.max_len_labels = max_len_labels
        self.ctc_loss_ratio = ctc_loss_ratio
        self.attn_loss_ratio = attn_loss_ratio
        self.ctc_loss_func = torch.nn.CTCLoss(zero_infinity=True)
        self.attn_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)


    def forward(self,data:dict,return_loss:bool,**kwargs):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def forward_train(self,data:dict,**kwargs):
        # assert  ("img","ctc_text","attn_text") in data.keys()
        visual_feature = data["img"]
        device = visual_feature.device
        batch_size = visual_feature.size(0)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ ctc_Refinement branch """
        ctc_prob = self.ctc_decoder(self.ctc_refine_fc(visual_feature.contiguous()))
        ctc_text = data["ctc_text"]
        ctc_length = data["ctc_length"]
        preds_size = torch.IntTensor([ctc_prob.size(1)] * batch_size).to(device)
        ctc_preds = ctc_prob.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc_loss_func(ctc_preds, ctc_text, preds_size, ctc_length.view(batch_size))

        """ attnblock_Refinement branch """
        attn_text = data["attn_text"]
        target = attn_text[:, 1:]# without [GO] Symbol
        attn_text =attn_text[:, :-1]# align with Attention.forward


        contextual_feature = visual_feature
        attn_loss= 0
        for index,scblock in enumerate(self.scblocks):
            contextual_feature, block_pred = scblock(visual_feature, contextual_feature, attn_text,True,
                                                              self.max_len_labels)
            attn_loss += self.attn_loss_func(block_pred.view(-1,block_pred.shape[-1]),
                                             target.contiguous().view(-1))
        return dict(loss_ctc=ctc_loss,loss_attn=attn_loss, loss=self.ctc_loss_ratio*ctc_loss+self.attn_loss_ratio*attn_loss)

        #计算loss

    def forward_test(self,data:dict,**kwargs):
        visual_feature = data["img"]
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        device = visual_feature.device
        batch_size = visual_feature.size(0)
        text_for_pred = torch.LongTensor(batch_size, self.max_len_labels + 1).fill_(0).to(device)
        contextual_feature = visual_feature
        for index,scblock in enumerate(self.scblocks):
            contextual_feature, block_pred = scblock(visual_feature, contextual_feature, text_for_pred,False,
                                                              self.max_len_labels)
        return block_pred

    def postprocess(self,block_preds:torch.Tensor):
        batch_size = block_preds.size(0)
        _, preds_index = block_preds.max(2)

        preds_str = self.converter.decode(preds_index, [self.max_len_labels] * batch_size)

        preds_prob = F.softmax(block_preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        pred_str_list= []
        confidence_score_list = []
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            pred_str_list.append(pred)
            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = float(pred_max_prob.cumprod(dim=0)[-1].cpu().numpy())
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
        return pred_str_list,confidence_score_list

class SelectiveContextualBlock(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,num_class:int):
        super(SelectiveContextualBlock, self).__init__()
        self.sequence_model = nn.Sequential(
            BidirectionalLSTM(input_size,hidden_size,hidden_size),
            BidirectionalLSTM(hidden_size,hidden_size,input_size)
        )
        self.sc_decoder = SelectiveDecoder(2*input_size,hidden_size,num_class)
    def forward(self,visual_feature:torch.Tensor,contextual_feature:torch.Tensor,attn_text_conver:torch.Tensor,is_train:bool=True,max_len_labels:int=50):
        contextual_feature = self.sequence_model(contextual_feature)
        fusion_feature = torch.cat((contextual_feature,visual_feature),2)
        block_pred = self.sc_decoder(fusion_feature,attn_text_conver,is_train,max_len_labels=max_len_labels)
        return contextual_feature,block_pred




class SelectiveDecoder(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,output_size:int):
        super(SelectiveDecoder, self).__init__()
        self.self_cv_attention_fc = nn.Linear(input_size,input_size)
        self.attention_decoder = Attention(input_size,hidden_size,output_size)
    def forward(self,fusion_feature:torch.Tensor,attn_text_conver:torch.Tensor,is_train:bool=True,max_len_labels:int=50):
        self_attention_map = self.self_cv_attention_fc(fusion_feature)
        fusion_feature = fusion_feature * self_attention_map
        decoder_probs = self.attention_decoder(fusion_feature,attn_text_conver,is_train,max_len_labels)
        return decoder_probs


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char:torch.Tensor, onehot_dim=38):
        device = input_char.device
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H:torch.Tensor, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        device = batch_H.device
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes
class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=True)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha





