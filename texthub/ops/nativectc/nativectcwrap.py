import torch
from torch.autograd import Function
from . import nativectc

class NativeCTCLossFunction(Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank=0,zero_infinity=True,
    ):
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity
        if not log_probs.is_cuda:
            raise NotImplementedError

        neg_log_likelihood, log_alpha, = nativectc.nativectc_forward(
            log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), blank,zero_infinity)
        if log_probs.requires_grad:
            ctx.save_for_backward(log_probs, targets, input_lengths,
                                  target_lengths, neg_log_likelihood, log_alpha)

        return neg_log_likelihood

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha = ctx.saved_tensors
        grad_log_probs = nativectc.nativectc_backward(
                grad_output, log_probs, targets, input_lengths.tolist(), target_lengths.tolist(),
                neg_log_likelihood, log_alpha,
                ctx.blank,ctx.zero_infinity
            )
        ##对forward的输入参数，如果不需要回传grad则返回None，参数个数一定要对齐

        return grad_log_probs, None, None, None,None,None

nativectc_loss = NativeCTCLossFunction.apply
