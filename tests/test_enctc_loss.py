
import os.path as osp
import os
import sys
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.modules.losses import NativeCTCcudaLoss,ENCTCcudaLoss



import torch
# Target are to be padded
T = 26      # Input sequence length
C = 5571      # Number of classes (including blank)
N = 64      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes
loss_fun = NativeCTCcudaLoss()
# loss_fun = ENCTCcudaLoss()
for i in range(20):
    
# Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_().cuda()

# Initialize random batch of targets (0 = blank, 1:C = classes)
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long).cuda()

    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).cuda()
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long).cuda()
# print(target_lengths)
# neg_log_likelihood, log_alpha,log_ent,log_alpha_ent = enctc_forward(
#             input, target, input_lengths, target_lengths, 0,True,0.2)
# ctc_loss = nn.CTCLoss()
# loss_fun = torch.nn.CTCLoss()
# ori_ctc = loss_fun(input, target, input_lengths, target_lengths)
# print(ori_ctc)
# print(input.shape,target.shape,input_lengths.shape,target_lengths.shape)
    loss = loss_fun(input, target, input_lengths, target_lengths)
    print(loss)
    loss.mean().backward()
