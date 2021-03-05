import torch
import torch.nn.functional as F


# @torch.jit.script
def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank: int = 0, reduction: str = 'none',
             alignment: bool = False):
    B = torch.arange(len(targets), device=input_lengths.device)
    targets_ = torch.cat([targets, targets[:, :1]], dim=-1)
    targets_ = torch.stack([torch.full_like(targets_, blank), targets_], dim=-1).flatten(start_dim=-2)
    diff_labels = torch.cat([torch.as_tensor([[False, False]], device=targets.device).expand(len(B), -1),
                             targets_[:, 2:] != targets_[:, :-2]], dim=1)

    # if the -inf is used as neutral element, custom logsumexp must be used
    # zero = float('-inf')
    # to avoid nan grad in torch.logsumexp
    zero = torch.finfo(log_probs.dtype).min

    zero, zero_padding = torch.tensor(zero, device=log_probs.device, dtype=log_probs.dtype), 2
    log_probs_ = log_probs.gather(-1, targets_.expand(len(log_probs), -1, -1))
    log_alpha = torch.full((len(log_probs), len(B), zero_padding + targets_.shape[-1]), zero, device=log_probs.device,
                           dtype=log_probs.dtype)
    log_alpha[0, :, zero_padding + 0] = log_probs[0, :, blank]
    log_alpha[0, :, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
    # log_alpha[1:, :, zero_padding:] = log_probs.gather(-1, targets_.expand(len(log_probs), -1, -1))[1:]
    for t in range(1, len(log_probs)):
        log_alpha[t, :, 2:] = log_probs_[t] + logadd(log_alpha[t - 1, :, 2:], log_alpha[t - 1, :, 1:-1],
                                                     torch.where(diff_labels, log_alpha[t - 1, :, :-2], zero))

    l1l2 = log_alpha[input_lengths - 1, B].gather(-1, torch.stack(
        [zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim=-1))
    loss = -torch.logsumexp(l1l2, dim=-1)
    if not alignment:
        return loss

    path = torch.zeros(len(log_alpha), len(B), device=log_alpha.device, dtype=torch.int64)
    path[input_lengths - 1, B] = zero_padding + 2 * target_lengths - 1 + l1l2.max(dim=-1).indices
    for t, indices in reversed(list(enumerate(path))[1:]):
        indices_ = torch.stack(
            [(indices - 2) * diff_labels[B, (indices - zero_padding).clamp(min=0)], (indices - 1).clamp(min=0),
             indices], dim=-1)
        path[t - 1] += (indices - 2 + log_alpha[t - 1, B].gather(-1, indices_).max(dim=-1).indices).clamp(min=0)
    return torch.zeros_like(log_alpha).scatter_(-1, path.unsqueeze(-1), 1.0)[..., (zero_padding + 1)::2]

def logadd(x0, x1, x2):
	# produces nan gradients in backward if -inf log-space zero element is used https://github.com/pytorch/pytorch/issues/31829
	return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)

