import  torch
import  torch.nn as nn
import  torch.nn.functional as F

from ..registry import HEADS
from .labelconverter import AttnLabelConverter


@HEADS.register_module
class Seq2SeqAttnHead(nn.Module):
    def __init__(self,input_size:int=512,hidden_size:int=256,charsets:str="", batch_max_length=25):
        super(Seq2SeqAttnHead, self).__init__()
        self.converter = AttnLabelConverter(charsets)
        """
        #在attentionlabelconvert 中,
        list_token = ['[GO]', '[s]']  
        list_character = list(charsets)
        self.character = list_token + list_character
        所以num_classes 应该为charset+2  "go,s"
        """
        self.num_classes = len(self.converter.character)
        self.batch_max_length = batch_max_length

        self.encoder = EncoderRNNImage(input_size=input_size,hidden_size=hidden_size,batch_max_length=batch_max_length)
        self.decoder = AttnDecoderRNN(hidden_size=hidden_size, num_classes=self.num_classes)

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO] token = ignore index 0

    def forward(self,data:dict,return_loss:bool,**kwargs):
        if return_loss:
            img_tensor = data["img"]
            enc_output, enc_hidden = self.encoder(img_tensor)
            targetTextTensor = data["label"]
            probs,target = self.decoder(enc_outputs=enc_output, hidden=enc_hidden,targeText=targetTextTensor)
            return probs,target
        else:
            img_tensor = data["img"]
            enc_output, enc_hidden = self.encoder(img_tensor)
            probs = self.decoder(enc_outputs=enc_output, hidden=enc_hidden, targeText=None)
            return probs

    def loss(self,probs:torch.Tensor,target:torch.Tensor):
        loss_recog = self.loss_func(probs.view(-1, probs.shape[-1]), target.contiguous().view(-1))
        return dict(loss_recog=loss_recog)

    def postprocess(self,data):
        """
        forward 得到输出结果,此处的data为probs tensor
        """
        batch_size = data.size(0)
        preds = data[:, :self.batch_max_length, :]
        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, [self.batch_max_length] * batch_size)
        return preds_str





class EncoderRNNImage(nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 256, batch_max_length=25):
        super(EncoderRNNImage, self).__init__()
        # Gru相比于lstm 没那么依赖上下文信息,ocr中上下文信息不足,所有采用gru
        self.rnn = nn.GRU(input_size, hidden_size=hidden_size, batch_first=True)
        # self.linear = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.batch_max_length = batch_max_length
    def forward(self, img_tensor: torch.Tensor):
        # 对输入的image_tensor进行encode,img_tensor 经过backbone抽完特征后[b, c, h, w] -> [b, w, c, h]->[b,w,c*h](默认)
        self.rnn.flatten_parameters()
        # batch_first=True
        # input (batch,seq_len, input_size)
        batch_size = img_tensor.size(0)
        device = img_tensor.device
        num_steps = self.batch_max_length + 1  # +1 for [s] at end of sentence.
        # 保存每一步的解码结果和隐状态
        enc_outputs = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        ##每个batch初始化一个隐状态
        # `(batchsize,num_layers * num_directions, hidden_size)`
        hidden = torch.FloatTensor(1, batch_size, self.hidden_size).fill_(0).to(device)
        for i in range(num_steps):
            # GRU  (input, h_0)
            # **input** of shape `(seq_len, batch, input_size)`
            # h_0 `(num_layers * num_directions, batch, hidden_size)`
            output, hidden = self.rnn(img_tensor[:, i, :].unsqueeze(1), hidden)
            # # 取hidden state作为encode 结果
            # output = self.linear(output)
            enc_outputs[:, i, :] = output.view(batch_size, self.hidden_size)
        return enc_outputs, hidden



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, num_classes):
        """
        output_size为字典大小,代表了每一个字符的概率
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = num_classes
        self.num_classes = num_classes
        self.attention_cell = AttentionCell(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.num_classes, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,enc_outputs: torch.Tensor, hidden: torch.Tensor, targeText: torch.Tensor=None):
        if targeText is not None:
            return self.forward_train(enc_outputs=enc_outputs,hidden=hidden,targeText=targeText)
        else:
            return self.forward_test(enc_outputs=enc_outputs,hidden=hidden)

    def forward_train(self, enc_outputs: torch.Tensor, hidden: torch.Tensor, targeText: torch.Tensor):
        """
        apply attention cell to decode the each step 's prob

        #decoder hiddent 作为query
        #enc_output 作为 context
        args:
            **enc_outputs** of shape `(batch, num_steps, hidden_size)`: tensor
          of encoder output for each steps
            **hidden** of shape `(1,batch,hidden_size)` encoder hidden state (1 bidirectional=Fasle)
            assume the encoder and decoder hidden_size are the same

            **targeText** shape `(batch,num_steps)` label encode tensor
        """
        targe = targeText[:,1:] # without [GO] Symbol
        textTensor = targeText[:,:-1]# align with Attention.forward
        device = enc_outputs.device
        batch_size, num_steps, _ = enc_outputs.size()

        decode_outputs = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        assert hidden.size(0) == 1 and hidden.size(2) == self.hidden_size
        decoder_hidden = hidden
        for i in range(num_steps):
            char_onehots = self._char_to_onehot(textTensor[:, i], onehot_dim=self.num_classes, device=device)
            # attent_context:[b,1,hidden]   ,query to shape  [batch size, output length, dimensions]
            attent_context, _ = self.attention_cell(decoder_hidden.view(batch_size,-1,self.hidden_size), context=enc_outputs)
            # 将char_onehot转化为hidden维度
            char_onehots = char_onehots.view(batch_size, 1, -1)
            ##此处cat,数据有点大
            concat_context = torch.cat([attent_context, char_onehots], 2)  # batch_size X (1)X(hidden+num_classes)
            output, decoder_hidden = self.gru(concat_context, decoder_hidden)
            decode_outputs[:, i, :] = output.view(batch_size, self.hidden_size)
        probs = self.out(decode_outputs)
        return probs,targe
    def forward_test(self,enc_outputs: torch.Tensor, hidden: torch.Tensor):
        device = enc_outputs.device
        batch_size, num_steps, _ = enc_outputs.size()
        probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)
        assert hidden.size(0) == 1 and hidden.size(2) == self.hidden_size
        decoder_hidden = hidden
        ##预测时 初始字符串为一个[go],然后根据上下文一步步decoder
        targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
        for i in range(num_steps):
            char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes, device=device)
            # attent_context:[b,1,hidden]   ,query to shape  [batch size, output length, dimensions]
            attent_context, _ = self.attention_cell(decoder_hidden.view(batch_size, -1, self.hidden_size),
                                                    context=enc_outputs)
            # 将char_onehot转化为hidden维度
            char_onehots = char_onehots.view(batch_size, 1, -1)
            ##此处cat,数据有点大
            concat_context = torch.cat([attent_context, char_onehots], 2)  # batch_size X (1)X(hidden+num_classes)
            output, decoder_hidden = self.gru(concat_context, decoder_hidden)
            prob_step = self.out(output.view(batch_size, self.hidden_size))
            probs[:, i, :] = prob_step
            _, next_input = prob_step.max(1)
            targets = next_input
        return probs  # batch_size x num_steps x num_classes
    def _char_to_onehot(self, input_char, device, onehot_dim: int):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot


class AttentionCell(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    Example:
         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """
    def __init__(self, dimensions, attention_type='general'):
        super(AttentionCell, self).__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)
        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)
        return output, attention_weights