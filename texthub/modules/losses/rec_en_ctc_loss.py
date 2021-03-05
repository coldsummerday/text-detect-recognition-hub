import  torch
import  torch.nn as nn
import numpy as np
from ..registry import LOSSES

@LOSSES.register_module
class EnCTCLoss(nn.Module):
    def __init__(self,blank = 0,reduction='sum',uni_rate=1.5,h_rate=0.2):
        """
        The pytorch-implementation of EN-CTC loss
        Args:
            blank (int, optional): blank label. Default :math:`0`.
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the output losses will be divided by the target lengths and
                then the mean over the batch is taken. Default: 'mean'
         Example::
            N, H, T, C = 16, 8, 32, 20
            ctc_loss = EnCTCLoss()
            pred = torch.randn(T, N, C).log_softmax(2).detach()
            target = torch.randint(1, C, (N, T), dtype=torch.long)
            input_lengths = torch.full((N,), T, dtype=torch.long,)
            target_lengths = torch.randint(10, 31, (N,), dtype=torch.long)
            loss = ctc_loss(pred, target, input_lengths, target_lengths)
            loss.backward()
        """
        super(EnCTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.h_rate = h_rate

        self.eps_nan = -1e8
        self.eps = 1e-6
        self.uni_rate=uni_rate

    def pre_make_mask(self,pred_length:torch.Tensor,target_length:torch.Tensor,Time:int):
        device = target_length.device
        batch = target_length.size(0)
        # target_max = target_length.max().item()
        # target = torch.zeros((batch, target_max)).to(device)
        uniform_mask = torch.zeros(Time, batch).type(torch.ByteTensor).to(device)
        for index, (target_size, size) in enumerate(zip(target_length, pred_length)):
            uni_length = int(self.uni_rate * (size.data.item() / (target_size.data.item() + 1)))
            uniform_mask[-uni_length:, index] = 1
        return uniform_mask

    def forward(self,pred:torch.Tensor,target:torch.Tensor,pred_length:torch.Tensor,target_length:torch.Tensor):
        """
        define:
        alpha(t, b, i) means the probability of end with output target i till time t
        beta(t, b, j, 2) means the probability of output only target j, from time t to now(用于等间距ctc，从t到现在只输出j的概率

        pred:Tensor of size:math:`(Time,Batch,voca_size+1)`
        pred_length:Tensor of size:math:`(Batch,)`
        target:Tensor of size :math:`(Batch, S)` S:max_target_length
        target_length:Tensor of size:math`(Batch)`
        uniform_mask:(Time,batch)

        alpha: (Time, batch, 2S+1) ∑p(π|x)
        beta: (Time, batch, 2S+1)  ∑p(π|x)logp(π|x)
        H: -beta/alpha+log(alpha)
        output:
        """

        device = pred.device
        T,B = pred.size(0),pred.size(1)
        max_target_length = target.size(1)

        has_equal = None
        uniform_mask = self.pre_make_mask(pred_length,target_length,T)


        # (Batch,S,2)  [:,:,0]是空白，[:,:,1]是原字符
        target_with_blank = torch.cat((torch.zeros(B,max_target_length,1).type(torch.LongTensor).to(device),target[:,:,None]),dim=2)


        pred_blank_p = pred[:,:,self.blank] #(time,batch)
        ## None 用于拓展维度torch.arange(0,T) [0,1,2...T],size = 1 拓展为[T,1,1,1]
        time_index = torch.arange(0,T).type(torch.LongTensor).to(device)[:,None,None,None]
        ##[1,Batch,1,1]
        batch_index = torch.arange(0,B).type(torch.LongTensor).to(device)[None,:,None,None]

        #(t,batch,s,2),(t,batch,s,0)为空白符的概率，1为s字符的概率
        pred = pred[time_index,batch_index,target_with_blank[None,:]]

        #batch,s-1
        # 用字符串[：-1]与[1：]去匹配，得到每个target字符串中是否有重复字符出现。[0,1,14,14,15],会在0，3匹配个true，所以target_equals为[：0]第几条数据，【：1】为重复字符的开始位置
        targets_equals = torch.nonzero(torch.eq(target[:,:-1],target[:,1:]),as_tuple=False)
        if len(targets_equals.size())==2:
            te_b = targets_equals[:,0]
            te_s = targets_equals[:,1]
            has_equal = True
        else:
            has_equal = False


        #初始化第0步的概率,
        betas = torch.ones((T,B,max_target_length,2),requires_grad=True).type(torch.FloatTensor).to(device)*self.eps_nan
        betas[0] = pred[0]

        ##最大熵
        beta_ent = torch.ones((T,B,max_target_length,2),requires_grad=True).type(torch.FloatTensor).to(device)*self.eps_nan
        beta_ent[0] = pred[0] +torch.log(-pred[0]+self.eps)

        # (1, batch, S)
        # alphas = T.cat((pred[0, :, 0, 1, None], T.ones(batch, U-1).type(floatX)*eps_nan), dim=1)[None]
        alphas = torch.ones((B,max_target_length),requires_grad=True).type(torch.FloatTensor).to(device)*self.eps_nan
        alphas[:,0] = pred[0,:,0,1]
        alphas = alphas.unsqueeze(0)


        alphas_ent = torch.ones((B,max_target_length),requires_grad=True).type(torch.FloatTensor).to(device)*self.eps_nan
        alphas_ent[:,0] = pred[0,:,0,1] + torch.log(-pred[0,:,0,1]+self.eps)
        alphas_ent = alphas_ent.unsqueeze(0)




        batch_range = torch.arange(0, B).to(device)
        ## prob of emit the last token till now

        ##label 为最后s最后一项的概率 (1,batch)
        labels = alphas[-1][batch_range,target_length-1].unsqueeze(0).clone().to(device) + (1-uniform_mask[0])*self.eps_nan
        labels_ent = alphas_ent[-1][batch_range,target_length-1].unsqueeze(0).clone().to(device) + (1-uniform_mask[0])*self.eps_nan


        for t in range(1,T):
            ##空白符+字符概率之和
            betas[:t,:,:,0] = betas[:t,:,:,0] + pred[t,None][:,:,:,0]
            betas[:t, :, :, 1] = log_sum_exp(betas[:t, :, :, 0], betas[:t, :, :, 1]) + pred[t,None][:,:,:,0]

            betas[t] = pred[t]

            ## 最大熵,log(空白符+s字符)+log(p)
            beta_ent[:t] = log_sum_exp(torch.cat((beta_ent[:t, :, :, 0, None], \
                                               log_sum_exp(beta_ent[:t, :, :, 0, None], \
                                                           beta_ent[:t, :, :, 1, None])), dim=-1) \
                                        + pred[t, None], \
                                        betas[:t].clone() + torch.log(-pred[t, None] + self.eps))
            beta_ent[t] = pred[t] + torch.log(-pred[t] + self.eps)


            alphas_t = torch.cat((betas[0, :, 0, 1][:, None].clone() + \
                              (1 - uniform_mask[-t, :, None]) * self.eps_nan, \
                              log_sum_exp_axis(alphas[:, :, :-1] + betas[1:t + 1, :, 1:, -1].clone(), \
                                               uniform_mask[-t:, :, None].expand(t, B, max_target_length - 1), dim=0)),
                             dim=1)
            alphas_t_ent = torch.cat((beta_ent[0, :, 0, 1][:, None].clone() + \
                                  (1 - uniform_mask[-t, :, None]) * self.eps_nan, \
                                  log_sum_exp_axis(
                                      log_sum_exp(alphas_ent[:, :, :-1] + betas[1:t + 1, :, 1:, -1].clone(), \
                                                  alphas[:, :, :-1] + beta_ent[1:t + 1, :, 1:, -1].clone()), \
                                      uniform_mask[-t:, :, None].expand(t, B, max_target_length - 1), dim=0)), dim=1)
            if has_equal and t!=1:
                alphas_t[te_b, 1 + te_s] = log_sum_exp_axis(alphas[:-1][:, te_b, te_s] \
                                                            + pred_blank_p[1:t][:, te_b] \
                                                            + betas[2:t + 1, :, :, -1][:, te_b, 1 + te_s].clone(),
                                                            uniform_mask[-t + 1:][:, te_b],
                                                            dim=0).clone() if t >= 2 else self.eps_nan
                alphas_t_ent[te_b, 1 + te_s] = log_sum_exp(
                    log_sum_exp_axis(pred_blank_p[1:t][:, te_b] + \
                                     log_sum_exp(alphas_ent[:-1][:, te_b, te_s] + betas[2:t + 1, :, :, -1][:, te_b,
                                                                                  1 + te_s].clone(), \
                                                 alphas[:-1][:, te_b, te_s] + beta_ent[2:t + 1, :, :, -1][:, te_b,
                                                                              1 + te_s].clone() \
                                                 ),
                                     uniform_mask[-t + 1:][:, te_b],
                                     dim=0).clone(), \
                    log_sum_exp_axis(alphas[:-1][:, te_b, te_s] \
                                     + pred_blank_p[1:t][:, te_b] \
                                     + torch.log(-pred_blank_p[1:t][:, te_b] + self.eps) \
                                     + betas[2:t + 1, :, :, -1][:, te_b, 1 + te_s].clone(),
                                     uniform_mask[-t + 1:][:, te_b],
                                     dim=0).clone()) if t >= 2 else self.eps_nan

            alphas = torch.cat((alphas, alphas_t[None, :]), dim=0)
            alphas_ent = torch.cat((alphas_ent, alphas_t_ent[None, :]), dim=0)
            labels_t = log_sum_exp(labels[-1] + pred_blank_p[t] + (1 - uniform_mask[t]) * self.eps_nan,
                                   alphas[-1][batch_range, target_length - 1])
            labels_t_ent = log_sum_exp(labels_ent[-1] + pred_blank_p[t] + (1 - uniform_mask[t]) * self.eps_nan, \
                                       labels[-1] + pred_blank_p[t] + torch.log(-pred_blank_p[t] + self.eps) + (
                                                   1 - uniform_mask[t]) * self.eps_nan, \
                                       alphas_ent[-1][batch_range, target_length - 1])
            
            labels = torch.cat((labels, labels_t[None]), dim=0).clone()
            labels_ent = torch.cat((labels_ent, labels_t_ent[None]), dim=0).clone()

        lt = labels[pred_length - 1, batch_range]
        lt_ent = labels_ent[pred_length - 1, batch_range]




        #entropy
        H = torch.exp(lt_ent - lt) + lt

        #ctc loss
        ori_loss = -lt
        loss = (1-self.h_rate)*ori_loss-self.h_rate*H
        if self.reduction == 'mean':
            return loss / target_length.type(torch.float)
        elif self.reduction=='sum':
            return loss.sum()


# def _logsumexp(a, b):
#     '''
#     np.log(np.exp(a) + np.exp(b))
#     '''
#
#     if a < b:
#         a, b = b, a
#
#     if b == -np.float('inf'):
#         return a
#     else:
#         return a + np.log(1 + np.exp(b - a))
#
# def logsumexp(*args):
#     '''
#     from scipy.special import logsumexp
#     logsumexp(args)
#     '''
#     res = args[0]
#     for e in args[1:]:
#         res = _logsumexp(res, e)
#     return res


def log_sum_exp_axis(a:torch.Tensor, uniform_mask=None, dim=0):
    device = a.device
    assert dim == 0
    eps_nan = -1e8
    eps = 1e-26
    _max = torch.max(a, dim=dim)[0]

    if not uniform_mask is None:

        nz_mask2 = (torch.gt(a, eps_nan) * uniform_mask).type(torch.bool)

        nz_mask1 = torch.gt(_max, eps_nan) * torch.ge(torch.max(uniform_mask, dim=dim)[0], 1).type(torch.bool)
    else:
        nz_mask2 = torch.gt(a, eps_nan)
        nz_mask1 = torch.gt(_max, eps_nan)


    # a-max
    a = a - _max[None]

    # exp
    _exp_a = torch.zeros_like(a).type(torch.FloatTensor).to(device)
    _exp_a[nz_mask2] = torch.exp(a[nz_mask2])

    # sum exp
    _sum_exp_a = torch.sum(_exp_a, dim=dim)

    out = torch.ones_like(_max).type(torch.FloatTensor).to(device) * eps_nan
    out[nz_mask1] = torch.log(_sum_exp_a[nz_mask1] + eps) + _max[nz_mask1]
    return out

def log_sum_exp(*arrs):
#    return T.max(a.clone(), b.clone()) + T.log1p(T.exp(-T.abs(a.clone()-b.clone())))
    c = torch.cat(list(map(lambda x:x[None], arrs)), dim=0)
    return log_sum_exp_axis(c, dim=0)


# if __name__ == '__main__':
#     N, H, T, C = 16, 8, 32, 20
#     ctc_loss = EnCTCLoss()
#     pred = torch.randn(T, N, C).log_softmax(2).detach()
#     target = torch.randint(1, C, (N, T), dtype=torch.long)
#     input_lengths = torch.full((N,), T, dtype=torch.long,)
#     target_lengths = torch.randint(10, 31, (N,), dtype=torch.long)
#     loss = ctc_loss(pred, target, input_lengths, target_lengths)
#     loss.backward()






