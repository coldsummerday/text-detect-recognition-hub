# coding=utf-8
import os
import torch as T
from torch.autograd import Variable
from ..registry import LOSSES


cuda = True
if cuda:
    floatX = T.cuda.FloatTensor
    #    floatX = T.cuda.DoubleTensor
    intX = T.cuda.IntTensor
    byteX = T.cuda.ByteTensor
    longX = T.cuda.LongTensor
else:
    floatX = T.FloatTensor
    intX = T.IntTensor
    byteX = T.ByteTensor
    longX = T.LongTensor

def log_sum_exp_axis(a, uniform_mask=None, dim=0):
    assert dim == 0
    eps_nan = -1e8
    eps = 1e-26
    _max = T.max(a, dim=dim)[0]

    if not uniform_mask is None:
        nz_mask2 =( T.gt(a, eps_nan) * uniform_mask).type(T.bool)
        nz_mask1 = (T.gt(_max, eps_nan) * T.ge(T.max(uniform_mask, dim=dim)[0], 1)).type(T.bool)
    else:
        nz_mask2 = T.gt(a, eps_nan)
        nz_mask1 = T.gt(_max, eps_nan)

    # a-max
    a = a - _max[None]

    # exp
    _exp_a = T.zeros_like(a).type(floatX)
    _exp_a[nz_mask2] = T.exp(a[nz_mask2])

    # sum exp
    _sum_exp_a = T.sum(_exp_a, dim=dim)

    out = T.ones_like(_max).type(floatX) * eps_nan
    out[nz_mask1] = T.log(_sum_exp_a[nz_mask1] + eps) + _max[nz_mask1]
    return out

def log_sum_exp(*arrs):
#    return T.max(a.clone(), b.clone()) + T.log1p(T.exp(-T.abs(a.clone()-b.clone())))
    c = T.cat(list(map(lambda x:x[None], arrs)), dim=0)
    return log_sum_exp_axis(c, dim=0)


@LOSSES.register_module
class EsEntCTCLoss(T.nn.Module):
    def __init__(self,uni_rate=1.5,h_rate=0.2,reduction="mean"):
        super(EsEntCTCLoss, self).__init__()
        self.uni_rate = uni_rate
        self.h_rate = h_rate
        self.reduction = reduction

    def forward(self,out, targets, sizes, target_sizes):
        H,ctc_loss = seg_ctc_ent_cost(out,targets,sizes,target_sizes)
        total_loss = (1 - self.h_rate) * ctc_loss - self.h_rate * H
        if self.reduction == 'mean':
            return total_loss / target_sizes.type(T.float)
        elif self.reduction == 'sum':
            return total_loss.sum()










def seg_ctc_ent_cost(out, targets, sizes, target_sizes, uni_rate=1.5):
    #    A batched version for uni_alpha_cost
    #    param out: (Time, batch, voca_size+1)
    #    param targets: targets without splited
    #    param sizes: size for out (N)
    #    param target_sizes: size for targets (N)
    '''
    out = out.cpu()
    targets = targets.cpu()
    sizes = sizes.cpu()
    target_sizes = target_sizes.cpu()
    '''

    Time = out.size(0)
    # pred = T.nn.functional.log_softmax(out, dim=-1)
    loss_func = seg_ctc_ent_loss_log

    offset = 0
    batch = target_sizes.size(0)
    uniform_mask = Variable(T.zeros(Time, batch).type(byteX))
    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        offset += target_size.item()
        uni_length = int(uni_rate * (size.data.item() / (target_size.data.item() + 1)))
        uniform_mask[-uni_length:, index] = 1

    if not cuda:
        H, costs = loss_func(out.cpu(), sizes.data.type(longX), targets, target_sizes.data.type(longX), uniform_mask)
    else:
        H, costs = loss_func(out, sizes.data.type(longX), targets, target_sizes.data.type(longX), uniform_mask)
    return H, costs




def seg_ctc_ent_loss_log(pred, pred_len, token, token_len, uniform_mask, blank=0):
    '''
    alpha(t, b, i) means the probability of end with output token i till time t
    beta(t, b, j, 2) means the probability of output only token j, from time t to now
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch,)
    :param token: (batch, U=token_len)
    :param token_len: (batch)
    :param blank: 0

    :out alpha: (Time, batch, 2U+1) ∑p(π|x)
    :out beta: (Time, batch, 2U+1)  ∑p(π|x)logp(π|x)
    :out H: -beta/alpha+log(alpha)
    :return: cost
    '''
    eps_nan = -1e8
    eps = 1e-6

    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)

    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2)  # (batch, U, 2)
    pred_blank = pred[:, :, 0]  # (Time, batch)
    pred = pred[T.arange(0, Time).type(longX)[:, None, None, None], T.arange(0, batch).type(longX)[None, :, None,
                                                                    None], token_with_blank[None, :]]
    # (Time, batch, U, 2)

    token_equals = T.nonzero(T.eq(token[:, :-1], token[:, 1:]),as_tuple=False)  # batch, U-1
    if len(token_equals.size()) == 2:
        te_b = token_equals[:, 0]
        te_u = token_equals[:, 1]
        have_equal = True
    else:
        have_equal = False

    betas = T.cat((pred[0, None], T.ones(Time - 1, batch, U, 2).type(floatX) * eps_nan), dim=0)
    betas_ent = T.cat(
        (pred[0, None] + T.log(-pred[0, None] + eps), T.ones(Time - 1, batch, U, 2).type(floatX) * eps_nan), dim=0)
    # (Time, batch, U, 2)
    alphas = T.cat((pred[0, :, 0, 1, None], T.ones(batch, U - 1).type(floatX) * eps_nan), dim=1)[None]
    alphas_ent = \
    T.cat((pred[0, :, 0, 1, None] + T.log(-pred[0, :, 0, 1, None] + eps), T.ones(batch, U - 1).type(floatX) * eps_nan),
          dim=1)[None]
    # (1, batch, U)
    batch_range = T.arange(0, batch).type(longX)
    labels = alphas[-1][batch_range, token_len - 1][None].clone() + \
             (1 - uniform_mask[0].type(floatX)) * eps_nan  # prob of emit the last token till now
    labels_ent = alphas_ent[-1][batch_range, token_len - 1][None].clone() + \
                 (1 - uniform_mask[0].type(floatX)) * eps_nan  # prob of emit the last token till now
    # (1, batch)

    for t in T.arange(1, Time).type(longX):
        betas[:t] = T.cat((betas[:t, :, :, 0, None], \
                           log_sum_exp(betas[:t, :, :, 0, None], betas[:t, :, :, 1, None])), dim=-1) \
                    + pred[t, None]
        betas[t] = pred[t]
        betas_ent[:t] = log_sum_exp(T.cat((betas_ent[:t, :, :, 0, None], \
                                           log_sum_exp(betas_ent[:t, :, :, 0, None], \
                                                       betas_ent[:t, :, :, 1, None])), dim=-1) \
                                    + pred[t, None], \
                                    betas[:t].clone() + T.log(-pred[t, None] + eps))
        betas_ent[t] = pred[t] + T.log(-pred[t] + eps)

        alphas_t = T.cat((betas[0, :, 0, 1][:, None].clone() + \
                          (1 - uniform_mask[-t, :, None].type(floatX)) * eps_nan, \
                          log_sum_exp_axis(alphas[:, :, :-1] + betas[1:t + 1, :, 1:, -1].clone(), \
                                           uniform_mask[-t:, :, None].expand(t.item(), batch, U - 1), dim=0)), dim=1)
        alphas_t_ent = T.cat((betas_ent[0, :, 0, 1][:, None].clone() + \
                              (1 - uniform_mask[-t, :, None].type(floatX)) * eps_nan, \
                              log_sum_exp_axis(log_sum_exp(alphas_ent[:, :, :-1] + betas[1:t + 1, :, 1:, -1].clone(), \
                                                           alphas[:, :, :-1] + betas_ent[1:t + 1, :, 1:, -1].clone()), \
                                               uniform_mask[-t:, :, None].expand(t.item(), batch, U - 1), dim=0)),
                             dim=1)
        if have_equal:
            alphas_t[te_b, 1 + te_u] = log_sum_exp_axis(alphas[:-1][:, te_b, te_u] \
                                                        + pred_blank[1:t][:, te_b] \
                                                        + betas[2:t + 1, :, :, -1][:, te_b, 1 + te_u].clone(),
                                                        uniform_mask[-t + 1:][:, te_b],
                                                        dim=0).clone() if t >= 2 else eps_nan
            alphas_t_ent[te_b, 1 + te_u] = log_sum_exp(
                log_sum_exp_axis(pred_blank[1:t][:, te_b] + \
                                 log_sum_exp(alphas_ent[:-1][:, te_b, te_u] + betas[2:t + 1, :, :, -1][:, te_b,
                                                                              1 + te_u].clone(), \
                                             alphas[:-1][:, te_b, te_u] + betas_ent[2:t + 1, :, :, -1][:, te_b,
                                                                          1 + te_u].clone() \
                                             ),
                                 uniform_mask[-t + 1:][:, te_b],
                                 dim=0).clone(), \
                log_sum_exp_axis(alphas[:-1][:, te_b, te_u] \
                                 + pred_blank[1:t][:, te_b] \
                                 + T.log(-pred_blank[1:t][:, te_b] + eps) \
                                 + betas[2:t + 1, :, :, -1][:, te_b, 1 + te_u].clone(),
                                 uniform_mask[-t + 1:][:, te_b],
                                 dim=0).clone()) if t >= 2 else eps_nan

        alphas = T.cat((alphas, alphas_t[None, :]), dim=0)
        alphas_ent = T.cat((alphas_ent, alphas_t_ent[None, :]), dim=0)
        labels_t = log_sum_exp(labels[-1] + pred_blank[t] + (1 - uniform_mask[t].type(floatX)) * eps_nan,
                               alphas[-1][batch_range, token_len - 1])
        labels_t_ent = log_sum_exp(labels_ent[-1] + pred_blank[t] + (1 - uniform_mask[t].type(floatX)) * eps_nan, \
                                   labels[-1] + pred_blank[t] + T.log(-pred_blank[t] + eps) + (
                                               1 - uniform_mask[t].type(floatX)) * eps_nan, \
                                   alphas_ent[-1][batch_range, token_len - 1])

        labels = T.cat((labels, labels_t[None]), dim=0).clone()
        labels_ent = T.cat((labels_ent, labels_t_ent[None]), dim=0).clone()

    lt = labels[pred_len - 1, batch_range]
    lt_ent = labels_ent[pred_len - 1, batch_range]

    H = T.exp(lt_ent - lt) + lt
    costs = -lt
    return H, costs  # (batch)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    import warnings

    warnings.filterwarnings('ignore')

    print('seg ctc env_________')
    test_seg_ctc(use_mine=True)
    # print('seg ctc_________')
    # test_seg_ctc(use_mine=False)
