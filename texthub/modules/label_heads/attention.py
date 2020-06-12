import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import HEADS
from .labelconverter import AttnLabelConverter

@HEADS.register_module
class AttentionHead(nn.Module):
    def __init__(self, input_size, hidden_size,charsets):
        super(AttentionHead, self).__init__()
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

    def forward(self,data:dict,return_loss,**kwargs):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def postprocess(self,data):
        return data


    def loss(self,probs,target):
        loss_recog = self.loss_func(probs.view(-1, probs.shape[-1]), target.contiguous().view(-1))
        return dict(loss_recog=loss_recog)

    def forward_train(self,data:dict,batch_max_length=25,**kwargs):
        img_tensor = data.get("img")
        device = img_tensor.device
        text = data["label"]
        target = text[:,1:] # without [GO] Symbol
        text = text[:,:-1]# align with Attention.forward

        batch_size = img_tensor.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

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
        return probs,target

    def forward_test(self,data:dict,batch_max_length=25,**kwargs):

        """
        在DataParallel 中,返回值有可能是map对象,需要单独处理

        :param batch_max_length:
        :param kwargs:
        :return:
        """
        img_tensor = data.get("img")
        batch_size = img_tensor.size(0)
        device = img_tensor.device
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))
        ##TODO:此处target 可能有问题
        targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
        probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(targets, device, onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(hidden, img_tensor.contiguous(), char_onehots)
            probs_step = self.generator(hidden[0])
            probs[:, i, :] = probs_step
            _, next_input = probs_step.max(1)
            targets = next_input

        preds = probs[:, :batch_max_length, :]
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, [batch_max_length]*batch_size)
        
        return preds_str

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
