import torch
from torch.autograd import Function
from . import enctc

class ENCTCLossFunction(Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths,ent_rate:float=0.05, blank=0,zero_infinity=True,
    ):
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity
        ctx.ent_rate = ent_rate
        if not log_probs.is_cuda:
            raise NotImplementedError

        neg_log_likelihood, log_alpha,log_ent,log_alpha_ent = enctc.enctc_forward(
            log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), blank,zero_infinity,ent_rate)

        if log_probs.requires_grad:
            ctx.save_for_backward(log_probs, targets, input_lengths,
                                  target_lengths, neg_log_likelihood, log_alpha,log_ent,log_alpha_ent)
        #TODO：计算ent_loss部分,应该有个logsum 部分  logsum(log_ent+neg_log_likelihood,-neg_log_likelihood)

        final_loss = neg_log_likelihood + ent_rate * (log_ent+neg_log_likelihood)
        # print(neg_log_likelihood,log_ent,log_alpha_ent)
        return final_loss

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha,log_ent,log_alpha_ent = ctx.saved_tensors
        grad_log_probs = enctc.enctc_backward(
                grad_output, log_probs, targets, input_lengths.tolist(), target_lengths.tolist(),
                neg_log_likelihood, log_alpha,log_ent,log_alpha_ent,
                ctx.blank,ctx.zero_infinity,ctx.ent_rate
            )
        ##对forward的输入参数，如果不需要回传grad则返回None，参数个数一定要对齐
        return grad_log_probs,None,None,None,None,None,None
        # return grad_log_probs, None, None, None, None

enctc_loss = ENCTCLossFunction.apply
