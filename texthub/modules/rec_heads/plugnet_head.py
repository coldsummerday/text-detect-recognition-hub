import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import HEADS
from .labelconverter import AttnLabelConverter
from ..backbones.rec_encoders import MeanShift
from ..backbones.rec_encoders import RCAN





@HEADS.register_module
class PlugAttentionHead(nn.Module):

    """
    batch_max_length  = img_w / 4
    """
    def __init__(self, input_size, hidden_size,charsets,sr_loss_ratio=0.01,batch_max_length=25):
        super(PlugAttentionHead, self).__init__()

        self.sr_loss_ratio = sr_loss_ratio
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)


        self.converter = AttnLabelConverter(charsets)
        """
        #在attentionlabelconvert 中,
        list_token = ['[GO]', '[s]']  
        list_character = list(charsets)
        self.character = list_token + list_character
        所以num_classes 应该为charset+2  "go,s"
        """
        num_classes = len(self.converter.character)
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=0) # ignore [GO] token = ignore index 0
        self.batch_max_length =batch_max_length


        self.loss_sr_func = torch.nn.L1Loss()
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)
        self.rcan_net = RCAN()


    def forward(self,data:dict,return_loss:bool):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def postprocess(self,preds:torch.Tensor):
        batch_size = preds.size(0)
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, [self.batch_max_length] * batch_size)
        scores = []
        return preds_str,scores



    def forward_train(self,data:dict):
        assert  "encoder_feats", "sharing_feats" in data.keys()
        assert "img" in data.keys()
        ## sr code
        sharing_feats = data.get("sharing_feats")
        fake_hr = self.rcan_net(sharing_feats)
        fake_hr = self.add_mean(fake_hr)
        hr_gt = data.get("img")
        loss_sr = self.loss_sr_func(fake_hr,hr_gt)





        ##attention_decoder
        img_tensor = data.get("encoder_feats")

        device = img_tensor.device
        text = data["label"]
        target = text[:,1:] # without [GO] Symbol
        text = text[:,:-1]# align with Attention.forward

        batch_size = img_tensor.size(0)
        num_steps = self.batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))


        for i in range(num_steps):
            # one-hot vectors for a i-th char. in a batch
            char_onehots = self._char_to_onehot(text[:, i], device=device,onehot_dim=self.num_classes)
            # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
            hidden, alpha = self.attention_cell(hidden, img_tensor, char_onehots)
            output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
        probs = self.generator(output_hiddens)

        loss_recog = self.loss_func(probs.view(-1, probs.shape[-1]), target.contiguous().view(-1))
        return dict(loss_recog=loss_recog,loss_sr=loss_sr,loss = loss_recog +self.sr_loss_ratio * loss_sr)


    def forward_test(self,data:dict,**kwargs):

        """
        在DataParallel 中,返回值有可能是map对象,需要单独处理

        :param batch_max_length:
        :param kwargs:
        :return:
        """
        img_tensor = data.get("encoder_feats")
        batch_size = img_tensor.size(0)
        device = img_tensor.device
        num_steps = self.batch_max_length + 1  # +1 for [s] at end of sentence.

        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
        probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(targets, device, onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(hidden, img_tensor.contiguous(), char_onehots)
            probs_step = self.generator(hidden[0])
            probs[:, i, :] = probs_step
            _, next_input = probs_step.max(1)
            targets = next_input
        preds = probs[:, :self.batch_max_length, :]
        return preds


    def _char_to_onehot(self, input_char,device, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot






    def ori_forward(self, batch_H, labels, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        device = batch_H.device
        text, length = self.converter.encode(labels, device,batch_max_length=batch_max_length)
        batch_size = batch_H.size(0)
        device = batch_H.device
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i],device,onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, device,onehot_dim=self.num_classes,device= device)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
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


