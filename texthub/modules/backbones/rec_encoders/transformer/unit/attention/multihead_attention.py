import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self,temperature:int,dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask=None)->(torch.Tensor,torch.Tensor):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        out = torch.matmul(attn, v)

        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,in_channels:int,k_channels:int,v_channels:int,n_head:int=8,dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.in_channles = in_channels
        self.k_channels = k_channels
        self.v_channels = v_channels
        self.n_head = n_head

        self.q_linear_layer = nn.Linear(in_features=in_channels,out_features=n_head * k_channels)
        self.k_linear_layer = nn.Linear(in_features=in_channels,out_features=n_head * k_channels)
        self.v_linear_layer = nn.Linear(in_features=in_channels,out_features=n_head * v_channels)

        self.attention_layer = DotProductAttention(temperature=k_channels**0.5,dropout=dropout)

        self.out_linear_layer = nn.Linear(in_features=n_head*v_channels,out_features=in_channels)
        self.drop_out_layer = nn.Dropout(p=dropout)

    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask=None):
        b,q_len,k_len,v_len = q.size(0),q.size(1),k.size(1),v.size(1)

        ##[b,t,c]->[b,t,head,q_channels]->[b,head,t,q_channels]
        q = self.q_linear_layer(q).view(b,q_len,self.n_head,self.k_channels).transpose(1,2)
        k = self.k_linear_layer(k).view(b,k_len,self.n_head,self.k_channels).transpose(1,2)
        v = self.v_linear_layer(v).view(b,v_len,self.n_head,self.v_channels).transpose(1,2)

        if mask!=None:
            mask = mask.unsqueeze(1)
        ##out:[b,len,n_head,v_channles]
        out,attn_weights = self.attention_layer(q,k,v,mask)
        ##[b,q_len,n_head* v_channels]
        out = out.transpose(1,2).contiguous().view(b,q_len,self.n_head*self.v_channels)
        out = self.out_linear_layer(out)
        out = self.drop_out_layer(out)

        return out,attn_weights




class GaussianSelfAttention(nn.Module):
    def __init__(self,num_heads:int=9,
                 hidden_size:int=256,
                 init_sigma_std:float=0.01,
                 init_mu_std:float=2.0,
                 attention_isotropic_gaussian = True,
                 max_width_height:int=100,
                 dropout = 0.1):
        super().__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size
        # assert config.hidden_size % config.num_attention_heads == 0, "num_attention_heads should divide hidden_size"
        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * hidden_size
        self.attention_isotropic_gaussian = attention_isotropic_gaussian

        # CAREFUL: if change something here, change also in reset_heads (TODO remove code duplication)
        # shift of the each gaussian per head
        self.attention_centers = nn.Parameter(
            torch.zeros(self.num_attention_heads, 2).normal_(0.0, init_mu_std)
        )

        if attention_isotropic_gaussian:
            # only one scalar (inverse standard deviation)
            # initialized to 1 + noise
            attention_spreads = 1 + torch.zeros(self.num_attention_heads).normal_(0, init_sigma_std)
        else:
            # Inverse standart deviation $Sigma^{-1/2}$
            # 2x2 matrix or a scalar per head
            # initialized to noisy identity matrix
            attention_spreads = torch.eye(2).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
            attention_spreads += torch.zeros_like(attention_spreads).normal_(0, init_sigma_std)

        self.attention_spreads = nn.Parameter(attention_spreads)

        self.value_layer = nn.Linear(self.all_head_size,hidden_size)

        # relative encoding grid (delta_x, delta_y, delta_x**2, delta_y**2, delta_x * delta_y)
        range_ = torch.arange(max_width_height)
        grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid([range_, range_])], dim=-1)
        relative_indices = grid.unsqueeze(0).unsqueeze(0) - grid.unsqueeze(-2).unsqueeze(-2)
        R = torch.cat([relative_indices, relative_indices ** 2, (relative_indices[..., 0] * relative_indices[..., 1]).unsqueeze(-1)], dim=-1)
        R = R.float()
        ##2d gausuion 位置区域
        self.register_buffer("R", R)
        self.dropout = nn.Dropout(dropout)

    def get_heads_target_vectors(self):
        if self.attention_isotropic_gaussian:
            a = c = self.attention_spreads ** 2
            b = torch.zeros_like(self.attention_spreads)
        else:
            # $\Sigma^{-1}$
            inv_covariance = torch.einsum('hij,hkj->hik', [self.attention_spreads, self.attention_spreads])
            a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]

        ##每次计算这里比较慢,根据attention_centers得到2d位置信息的value值，可以优化
        mu_1, mu_2 = self.attention_centers[:, 0], self.attention_centers[:, 1]

        t_h = -1/2 * torch.stack([
            -2*(a*mu_1 + b*mu_2),
            -2*(c*mu_2 + b*mu_1),
            a,
            c,
            2 * b
        ], dim=-1)
        return t_h

    def get_attention_probs(self, width:int, height:int):
        """Compute the positional attention for an image of size width x height
        Returns: tensor of attention probabilities (width, height, num_head, width, height)
        """
        u = self.get_heads_target_vectors()

        # Compute attention map for each head
        # d=5 与d相乘然后得到权重，分配到[w,h,heads,w,h]上. w,h 做  softmax
        attention_scores = torch.einsum('ijkld,hd->ijhkl', [self.R[:width,:height,:width,:height,:], u])
        # Softmax
        attention_probs = torch.nn.functional.softmax(attention_scores.view(width, height, self.num_attention_heads, -1),dim=-1)
        # attention_probs = torch.nn.Softmax(dim=-1)(attention_scores.view(width, height, self.num_attention_heads, -1))

        attention_probs = attention_probs.view(width, height, self.num_attention_heads, width, height)
        return attention_probs

    # def blured_attention(self, X):
    #     """Compute the weighted average according to gaussian attention without
    #     computing explicitly the attention coefficients.
    #
    #     Args:
    #         X (tensor): shape (batch, width, height, dim)
    #     Output:
    #         shape (batch, width, height, dim x num_heads)
    #     """
    #     num_heads = self.attention_centers.shape[0]
    #     batch, width, height, d_total = X.shape
    #     Y = X.permute(0, 3, 1, 2).contiguous()
    #
    #     kernels = []
    #     kernel_width = kernel_height = 7
    #     assert kernel_width % 2 == 1 and kernel_height % 2 == 1, 'kernel size should be odd'
    #
    #     for mean, std_inv in zip(self.attention_centers, self.attention_spreads):
    #         conv_weights = gaussian_kernel_2d(mean, std_inv, size=(kernel_width, kernel_height))
    #         conv_weights = conv_weights.view(1, 1, kernel_width, kernel_height).repeat(d_total, 1, 1, 1)
    #         kernels.append(conv_weights)
    #
    #     weights = torch.cat(kernels)
    #
    #     padding_width = (kernel_width - 1) // 2
    #     padding_height = (kernel_height - 1) // 2
    #     out = F.conv2d(Y, weights, groups=d_total, padding=(padding_width, padding_height))
    #
    #     # renormalize for padding
    #     all_one_input = torch.ones(1, d_total, width, height, device=X.device)
    #     normalizer = F.conv2d(all_one_input, weights,  groups=d_total, padding=(padding_width, padding_height))
    #     out /= normalizer
    #
    #     return out.permute(0, 2, 3, 1).contiguous()

    def forward(self, hidden_states:torch.Tensor):

        #
        # assert len(hidden_states.shape) == 4
        # b, w, h, c = hidden_states.shape

        b,c,h,w = hidden_states.shape
        ##[b,c,h,w]->[b,w,h,c]
        hidden_states = hidden_states.permute(0,3,2,1)


        attention_probs = self.get_attention_probs(w, h)
        attention_probs = self.dropout(attention_probs)

        #[w,h,heads,w,h] 后面两个[w,h]作为2D权重 ，乘输入 [b,w,h,c]->[b,w,h,heads,dim] 即[w,h]*[w,h]得到一个值作为乘attention后的激活值
        input_values = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_states)
        input_values = input_values.contiguous().view(b, w, h, -1)

        # if not self.attention_gaussian_blur_trick:
        #     attention_probs = self.get_attention_probs(w, h)
        #     attention_probs = self.dropout(attention_probs)
        #
        #     input_values = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_states)
        #     input_values = input_values.contiguous().view(b, w, h, -1)
        # else:
        #     input_values = self.blured_attention(hidden_states)

        output_value = self.value_layer(input_values)

        #[b,w,h,c]->[b,c,h,w]
        output_value = output_value.permute(0,3,2,1)

        return output_value, attention_probs






