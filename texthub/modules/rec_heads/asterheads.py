import torch
import torch.nn as nn
import torch.nn.functional as F
from .labelconverter import AttnLabelConverter
from ..registry import HEADS

"""
aster encoder = resnet + lstm
"""

@HEADS.register_module
class AsterAttentionRecognitionHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, attention_dim, charsets: str, beam_width=5,max_len_labels=25):
        super(AsterAttentionRecognitionHead, self).__init__()
        self.converter = AttnLabelConverter(charsets)
        """
        #在attentionlabelconvert 中,
        list_token = ['[GO]', '[s]']  
        list_character = list(charsets)
        self.character = list_token + list_character
        所以num_classes 应该为charset+2  "go,s"
        """
        self.num_classes = len(self.converter.character)
        self.max_len_labels = max_len_labels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.beam_width = beam_width
        self.decoder = DecoderUnit(hidden_dim=hidden_dim, input_dim=input_dim, num_classes=self.num_classes,
                                   attention_dim=attention_dim)

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO] token = ignore index 0

    def forward(self,data:dict,return_loss:bool,**kwargs):
        if return_loss==True:
            return self.forward_train(data=data)
        else:
            #TODO:forward use pro instead of beam_search
            return self.forward_test(data=data)



    def loss(self,probs:torch.Tensor,target:torch.Tensor):
        loss_recog = self.loss_func(probs.view(-1, probs.shape[-1]), target.contiguous().view(-1))
        return dict(loss_recog=loss_recog)

    def forward_test(self,data:dict):
        encoder_feats = data.get('img').contiguous()
        device = encoder_feats.device
        batch_size = encoder_feats.size(0)
        decoder_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        output_tensor = torch.FloatTensor(batch_size,self.max_len_labels+1,self.num_classes).fill_(0).to(device)
        ## +1 for [s] at end of sentence.
        for i in range(self.max_len_labels+1):
            if i==0:
                y_prev = torch.zeros((batch_size)).to(device)
            else:
                _, predicted = output.max(1)
                y_prev = predicted
            output,state = self.decoder(encoder_feats,decoder_state,y_prev)
            output_tensor[:,i,:] = output
        _, preds_index = output_tensor.max(2)
        return preds_index

    def postprocess(self,data:torch.Tensor):
        """
        pred_rob to pred_str
        """
        batch_size,seq_len = data.shape
        rec_pred = to_numpy(data)
        preds_str = self.converter.decode(rec_pred, [self.max_len_labels] * batch_size)
        return preds_str

    def forward_train(self, data: dict):
        encoder_feats = data.get('img').contiguous()

        device = encoder_feats.device

        rec_targets = data["label"][:,1:] # without [GO] Symbol
        batch_size = encoder_feats.size(0)
        decoder_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)


        # outputs = []

        output_tensor = torch.FloatTensor(batch_size,self.max_len_labels+1,self.num_classes).fill_(0).to(device)
        ##TODO:output 需要用torch 先定义好矩阵再存放，用list太慢了
        ## +1 for [s] at end of sentence.

        for i in range(self.max_len_labels+1):
            if i==0:
                y_prev = torch.zeros((batch_size)).fill_(0).to(device)
            else:
                y_prev = rec_targets[:,i-1]
            output,state = self.decoder(encoder_feats,decoder_state,y_prev)
            # outputs.append(output)
            output_tensor[:,i,:] = output
        # outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        # return outputs
        return output_tensor
    def beam_search(self,data:dict):
        encoder_feats = data.get('img').contiguous()
        device = encoder_feats.device
        beam_width = self.beam_width
        # eos =  '[s]'
        eos = self.converter.dict['[s]']

        def _inflate(tensor,times,dim):
            ##所有维度都重复一遍
            repeat_dims = [1] * tensor.dim()
            #在需要的维度重复times
            repeat_dims[dim] = times
            return tensor.repeat(*repeat_dims)

        # https://github.com/IBM/pytorch-seq2seq/blob/fede87655ddce6c94b38886089e05321dc9802af/seq2seq/models/TopKDecoder.py

        batch_size, seq_len, seq_dim = encoder_feats.size()
        # inflated_encoder_feats = _inflate(encoder_feats, beam_width, 0) # ABC --> AABBCC -/-> ABCABC
        inflated_encoder_feats = encoder_feats.unsqueeze(1)\
            .permute((1, 0, 2, 3)).repeat((beam_width, 1, 1, 1)).permute(
            (1, 0, 2, 3)).contiguous().view(-1, seq_len, seq_dim)
        # Initialize the decoder
        decoder_state = torch.zeros(1, batch_size * beam_width, self.hidden_dim).to(device)
        pos_index = (torch.Tensor(range(batch_size)) * beam_width).long().view(-1, 1).to(device)

        # Initialize the scores
        sequence_scores = torch.Tensor(batch_size * beam_width, 1).to(device)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.Tensor([i * beam_width for i in range(0, batch_size)]).long().to(device), 0.0)
        # sequence_scores.fill_(0.0)

        # Initialize the input vector
        y_prev = torch.zeros((batch_size * beam_width)).fill_(self.num_classes).to(device)

        # Store decisions for backtracking
        #(T,batch * beam_width,1)
        stored_scores_tensor = torch.FloatTensor(self.max_len_labels+1,batch_size*beam_width,1).fill_(0).to(device)
        stored_predecessors_tensor = torch.LongTensor(self.max_len_labels+1,batch_size * beam_width, 1).fill_(0).to(device)
        stored_emitted_symbols_tensor = torch.LongTensor(self.max_len_labels+1,batch_size*beam_width).fill_(0).to(device)

        # stored_scores = list()
        # stored_predecessors = list()
        # stored_emitted_symbols = list()

        # +1 for [s] at end of sentence.
        for i in range(self.max_len_labels+1):
            output,decoder_state = self.decoder(inflated_encoder_feats,decoder_state,y_prev)
            log_softmax_output = F.log_softmax(output,dim=1)
            sequence_scores = _inflate(sequence_scores,self.num_classes,1)
            sequence_scores += log_softmax_output

            #每一步选topk个
            scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_width, dim=1)
            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            y_prev = (candidates % self.num_classes).view(batch_size*beam_width)
            sequence_scores = scores.view(batch_size * beam_width,1)

            # Update fields for next timestep
            #torch 1.5.1 floor_divide
            # predecessors = (candidates.floor_divide(self.num_classes)  + pos_index.expand_as(candidates)).view(batch_size* beam_width,1)
            # torch 1.3.1 /
            predecessors = (candidates / self.num_classes + pos_index.expand_as(candidates)).view(
                batch_size * beam_width, 1)

            decoder_state = decoder_state.index_select(1,predecessors.squeeze())

            # Update sequence socres and erase scores for <eos> symbol so that they aren't expanded
            # stored_scores.append(sequence_scores.clone())
            stored_scores_tensor[i] = sequence_scores.clone()
            eos_indices = y_prev.view(-1, 1).eq(eos)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            # stored_predecessors.append(predecessors)
            stored_predecessors_tensor[i] = predecessors
            stored_emitted_symbols_tensor[i] = y_prev

            # stored_emitted_symbols.append(y_prev)
        # Do backtracking to return the optimal values
        # ====== backtrak ======#
        # Initialize return variables given different types
        #反向搜索
        p = []
        l = [[self.max_len_labels]*beam_width for _ in range(batch_size)]
        # the last step output of the beams are not sorted
        # thus they are sorted here
        # sorted_score,sorted_idx = stored_scores[-1].view(batch_size,beam_width).topk(beam_width)
        # s = sorted_score.clone()
        sorted_score,sorted_idx = stored_scores_tensor[-1].view(batch_size,beam_width).topk(beam_width)
        s = sorted_score.clone()


        batch_eos_found = [0] *batch_size

        # +1 for [s] at end of sentence.
        #t = self.max_len_labels -1
        t = self.max_len_labels
        t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(batch_size * beam_width)
        while t >= 0:
            # Re-order the variables with the back pointer
            # current_symbol = stored_emitted_symbols[t].index_select(0,t_predecessors)
            # t_predecessors = stored_predecessors[t].index_select(0,t_predecessors).squeeze()
            # eos_indices = stored_emitted_symbols[t].eq(eos).nonzero()
            current_symbol = stored_emitted_symbols_tensor[t].index_select(0,t_predecessors)
            t_predecessors = stored_predecessors_tensor[t].index_select(0,t_predecessors).squeeze()
            eos_indices = stored_emitted_symbols_tensor[t].eq(eos).nonzero()
            if eos_indices.dim() > 0 :
                for i in range(eos_indices.size(0)-1,-1,-1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_index = int(idx[0]/beam_width)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_width - (batch_eos_found[b_index] % beam_width) -1
                    batch_eos_found[b_index] +=1
                    res_idx = b_index * beam_width + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = stored_predecessors_tensor[t][idx[0]]
                    current_symbol[res_idx] = stored_emitted_symbols_tensor[t][idx[0]]
                    s[b_index,res_k_idx] = stored_scores_tensor[t][idx[0],[0]]
                    l[b_index][res_k_idx] = t + 1
            # record the back tracked results
            p.append(current_symbol)
            t-=1
        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s,re_sorted_idx = s.topk(beam_width)
        for b_index in range(batch_size):
            l[b_index] = [l[b_index][k_idx.item()] for k_idx in re_sorted_idx[b_index,:]]
        re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(batch_size * beam_width)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        p = [step.index_select(0,re_sorted_idx).view(batch_size,beam_width,-1)  for step in reversed(p)]
        rec_pred = torch.cat(p, -1)[:, 0, :]
        return rec_pred






class MLPAttentionCell(nn.Module):
    def __init__(self, input_dim, decoder_dim=256, attDim=256):
        super(MLPAttentionCell, self).__init__()
        self.decoder_dim = decoder_dim
        self.input_dim = input_dim
        self.attDim = attDim
        self.sEmbed = nn.Linear(decoder_dim, attDim)
        self.xEmbed = nn.Linear(input_dim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

        # self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.sEmbed.weight, std=0.01)
        nn.init.constant_(self.sEmbed.bias, 0)
        nn.init.normal_(self.xEmbed.weight, std=0.01)
        nn.init.constant_(self.xEmbed.bias, 0)
        nn.init.normal_(self.wEmbed.weight, std=0.01)
        nn.init.constant_(self.wEmbed.bias, 0)

    def forward(self, x, sPrev):
        batch_size, T, _ = x.size()  # [b x T x xDim]
        x = x.view(-1, self.input_dim)  # [(b x T) x xDim]
        xProj = self.xEmbed(x)  # [(b x T) x attDim]
        xProj = xProj.view(batch_size, T, -1)  # [b x T x attDim]

        sPrev = sPrev.squeeze(0)
        sProj = self.sEmbed(sPrev)  # [b x attDim]
        sProj = torch.unsqueeze(sProj, 1)  # [b x 1 x attDim]
        sProj = sProj.expand(batch_size, T, self.attDim)  # [b x T x attDim]

        sumTanh = torch.tanh(sProj + xProj)
        sumTanh = sumTanh.view(-1, self.attDim)

        vProj = self.wEmbed(sumTanh)  # [(b x T) x 1]
        vProj = vProj.view(batch_size, T)

        alpha = F.softmax(vProj, dim=1)  # attention weights for each sample in the minibatch
        return alpha


class DecoderUnit(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_classes, attention_dim):
        """
        self.decoder = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=num_classes, attDim=attDim)
        """
        super(DecoderUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.emd_dim = attention_dim

        self.attention_unit = MLPAttentionCell(input_dim, hidden_dim, attention_dim)
        self.target_embedding = nn.Embedding(num_classes + 1, self.emd_dim)  # the last is used for <BOS>
        self.gru = nn.GRU(input_size=input_dim + self.emd_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.target_embedding.weight, std=0.01)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, sPrev, yPrev):
        # x: feature sequence from the image decoder.
        batch_size, T, _ = x.size()
        alpha = self.attention_unit(x, sPrev)
        context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
        yProj = self.target_embedding(yPrev.long())

        self.gru.flatten_parameters()
        output, state = self.gru(torch.cat([yProj, context], 1).unsqueeze(1), sPrev)
        output = output.squeeze(1)
        output = self.fc(output)
        return output, state


class SequenceCrossEntropyLoss(nn.Module):
  def __init__(self,
               weight=None,
               size_average=True,
               ignore_index=-100,
               sequence_normalize=False,
               sample_normalize=True):
    super(SequenceCrossEntropyLoss, self).__init__()
    self.weight = weight
    self.size_average = size_average
    self.ignore_index = ignore_index
    self.sequence_normalize = sequence_normalize
    self.sample_normalize = sample_normalize

    assert (sequence_normalize and sample_normalize) == False

  def forward(self, input, target, length):
    _assert_no_grad(target)
    # length to mask
    batch_size, def_max_length = target.size(0), target.size(1)
    mask = torch.zeros(batch_size, def_max_length)
    for i in range(batch_size):
      mask[i,:length[i]].fill_(1)
    mask = mask.type_as(input)
    # truncate to the same size
    max_length = max(length)
    assert max_length == input.size(1)
    target = target[:, :max_length]
    mask =  mask[:, :max_length]
    input = to_contiguous(input).view(-1, input.size(2))
    input = F.log_softmax(input, dim=1)
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    output = - input.gather(1, target.long()) * mask
    # if self.size_average:
    #   output = torch.sum(output) / torch.sum(mask)
    # elif self.reduce:
    #   output = torch.sum(output)
    ##
    output = torch.sum(output)
    if self.sequence_normalize:
      output = output / torch.sum(mask)
    if self.sample_normalize:
      output = output / batch_size

    return output

def to_contiguous(tensor):
  if tensor.is_contiguous():
    return tensor
  else:
    return tensor.contiguous()

def _assert_no_grad(variable):
  assert not variable.requires_grad, \
    "nn criterions don't compute the gradient w.r.t. targets - please " \
    "mark these variables as not requiring gradients"

def to_numpy(tensor):
  if torch.is_tensor(tensor):
    return tensor.cpu().numpy()
  elif type(tensor).__module__ != 'numpy':
    raise ValueError("Cannot convert {} to numpy array"
                     .format(type(tensor)))
  return tensor