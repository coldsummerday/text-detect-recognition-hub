


# from .enctcwrapper import enctc_loss
# from .enctc import enctc_forward,enctc_backward

from texthub.ops.en_ctc import enctc_loss
import torch
# Target are to be padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_().cuda()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long).cuda()

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).cuda()
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long).cuda()

# neg_log_likelihood, log_alpha,log_ent,log_alpha_ent = enctc_forward(
#             input, target, input_lengths, target_lengths, 0,True,0.2)
# ctc_loss = nn.CTCLoss()
loss_fun = torch.nn.CTCLoss()
ori_ctc = loss_fun(input, target, input_lengths, target_lengths)
print(ori_ctc)
loss = enctc_loss(input, target, input_lengths, target_lengths,0.2,0,True)
print()
loss.sum().backward()
# >>> loss.backward()
# >>>
# >>>
# >>> # Target are to be un-padded
# >>> T = 50      # Input sequence length
# >>> C = 20      # Number of classes (including blank)
# >>> N = 16      # Batch size
# >>>
# >>> # Initialize random batch of input vectors, for *size = (T,N,C)
# >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# >>>
# >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
# >>> target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
# >>> target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
# >>> ctc_loss = nn.CTCLoss()
# >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
# >>> loss.backward()