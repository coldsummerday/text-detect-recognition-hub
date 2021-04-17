import  torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Learned2DRelativeSelfAttention(nn.Module):
    def __init__(self,
                 num_heads:int=9,
                 hidden_size:int=256,
                 max_position_embeddings:int=16,
                 dropout:float=0.1):
        super(Learned2DRelativeSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.all_head_size = hidden_size * self.num_heads

        position_embedding_size = hidden_size // 2
        self.width_embeddings_layer = nn.Embedding(2 * max_position_embeddings , position_embedding_size)
        self.height_embeddings_layer = nn.Embedding(2 * max_position_embeddings , position_embedding_size)

        self.head_keys_height_layer = nn.Linear(position_embedding_size, self.num_heads, bias=False)
        self.head_keys_width_layer = nn.Linear(position_embedding_size, self.num_heads, bias=False)

        # self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        # self.key_layer = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.final_value_layer = nn.Linear(self.all_head_size,hidden_size)



        deltas = torch.arange(max_position_embeddings).view(1, -1) - torch.arange(max_position_embeddings).view(-1, 1)
        # shift the delta to [0, 2 * max_position_embeddings - 1]
        relative_indices = deltas + max_position_embeddings - 1
        self.register_buffer("relative_indices_tensor", relative_indices)
        ##相对位置tensor  100,[]
        """
        tensor([[ 99, 100, 101,  ..., 196, 197, 198],
        [ 98,  99, 100,  ..., 195, 196, 197],
        [ 97,  98,  99,  ..., 194, 195, 196],
        ...,
        [  2,   3,   4,  ...,  99, 100, 101],
        [  1,   2,   3,  ...,  98,  99, 100],
        [  0,   1,   2,  ...,  97,  98,  99]])
        """

    def get_attention_probs(self,height:int, width:int):
        """LEGACY
        Compute the positional attention for an image of size width x height
        Returns: tensor of attention probabilities (height,width,num_heads,height,width)
        """
        width_relative_indices = self.relative_indices_tensor[:width, :width].reshape(-1)

        width_scores = self.head_keys_width_layer(self.width_embeddings_layer(width_relative_indices)).view(1, width, 1, width,
                                                                                    self.num_heads)

        height_relative_indices = self.relative_indices_tensor[:height, :height].reshape(-1)
        height_scores = self.head_keys_height_layer(self.height_embeddings_layer(height_relative_indices)).view(height, 1, height, 1,
                                                                                    self.num_heads)

        # -- H, W, H, W, num_attention_heads
        attention_scores = width_scores + height_scores
        # -- H, W, num_attention_heads, H, W
        attention_scores = attention_scores.permute(0, 1, 4, 2, 3)

        # -- H, W, num_attention_heads, H, W
        # flatten_shape = [height, width, self.num_heads, height * width]
        # unflatten_shape = [height, width, self.num_heads, height, width]
        # attention_probs = nn.Softmax(dim=-1)(attention_scores.view(*flatten_shape)).view(*unflatten_shape)


        attention_probs = torch.nn.functional.softmax(
            attention_scores.view(height, width, self.num_heads, height * width), dim=-1).view(height, width,
                                                                                               self.num_heads, height,
                                                                                               width)
        return attention_probs

    def forward(self, hidden_states:torch.Tensor):
        assert len(hidden_states.shape) == 4
        b,c,h,w = hidden_states.shape
        sqrt_normalizer = math.sqrt(self.hidden_size)

        #(height, width, num_heads, height, width)
        attention_probs = self.get_attention_probs(height=h,width=w) /sqrt_normalizer
        attention_probs = self.dropout(attention_probs)

        input_values = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_states.permute(0,2,3,1))
        input_values = input_values.contiguous().view(b, h, w, -1)
        output_value = self.final_value_layer(input_values)
        #[b,h,w,c]->[b,c,h,w]
        return output_value.permute(0,3,1,2),attention_probs





    # def compute_attention_scores(self, hidden_states:torch.Tensor):
    #     """Compute the positional attention for an image of size width x height
    #     Returns: tensor of attention scores (1 or batch, width, height, num_head, width, height)
    #
    #     Attention scores:
    #         * Position only
    #             Options: use_attention_data=False, query_positional_score=False
    #             w_q^T * r
    #             where w_q is a learned vector per head
    #         * Query and positional encoding (without query key attention scores),
    #             same as q * r in (Ramachandran et al., 2019)
    #             Options: use_attention_data=False, query_positional_score=True
    #             X * W_Q * r
    #         * With data
    #             same as q*k + q*r in (Ramachandran et al., 2019)
    #             Options: use_attention_data=True, query_positional_score=True
    #             X * W_Q * W_K^T * X^T + X * W_Q * r
    #         * Last option use_attention_data=True, query_positional_score=False was not used
    #     """
    #     batch_size, height, width, hidden_dim = hidden_states.shape
    #
    #     # compute query data if needed
    #     if self.use_attention_data or self.query_positional_score:
    #         q = self.query(hidden_states)
    #         q = q.view(batch_size, width, height, self.num_attention_heads, self.hidden_size)
    #
    #     # compute key data if needed
    #     if self.use_attention_data:
    #         k = self.key(hidden_states)
    #         k = k.view(batch_size, width, height, self.num_attention_heads, self.hidden_size)
    #
    #     # Compute attention scores based on position
    #     # Probably not optimal way to order computation
    #     relative_indices = self.relative_indices[:width, :width].reshape(-1)
    #     row_embeddings = self.row_embeddings(relative_indices)
    #
    #     relative_indices = self.relative_indices[:height, :height].reshape(-1)
    #     col_embeddings = self.col_embeddings(relative_indices)
    #
    #     # keep attention scores/prob for plotting
    #     attention_scores_per_type = {}
    #     sqrt_normalizer = math.sqrt(self.hidden_size)
    #
    #     if not self.query_positional_score:
    #         # Caveat: sqrt rescaling is not used in this case
    #         row_scores = self.head_keys_row(row_embeddings).view(1, width, 1, width, self.num_attention_heads)
    #         col_scores = self.head_keys_col(col_embeddings).view(height, 1, height, 1, self.num_attention_heads)
    #         # -- H, W, H, W, num_attention_heads
    #         attention_scores = row_scores + col_scores
    #         # -- H, W, num_attention_heads, H, W
    #         attention_scores = attention_scores.permute(0, 1, 4, 2, 3)
    #         # -- 1, H, W, num_attention_heads, H, W
    #         attention_scores = attention_scores.unsqueeze(0)
    #
    #         attention_scores_per_type["w_q^Tr"] = attention_scores
    #
    #     else:  # query_positional_score
    #         # B, W, H, num_attention_heads, D // 2
    #         q_row = q[:, :, :, :, :self.hidden_size // 2]
    #         q_col = q[:, :, :, :, self.hidden_size // 2:]
    #
    #         row_scores = torch.einsum("bijhd,ikd->bijhk", q_row, row_embeddings.view(width, width, -1))
    #         col_scores = torch.einsum("bijhd,jld->bijhl", q_col, col_embeddings.view(height, height, -1))
    #
    #         # -- B, H, W, num_attention_heads, H, W
    #         attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)
    #         attention_scores = attention_scores / sqrt_normalizer
    #
    #         # save
    #         attention_scores_per_type["q^Tr"] = attention_scores
    #
    #     # Compute attention scores based on data
    #     if self.use_attention_data:
    #         attention_content_scores = torch.einsum("bijhd,bklhd->bijhkl", q, k)
    #         attention_content_scores = attention_content_scores / sqrt_normalizer
    #         attention_scores = attention_scores + attention_content_scores
    #
    #         # save
    #         attention_scores_per_type["q^Tk"] = attention_content_scores
    #
    #     return attention_scores, attention_scores_per_type


# class Learned2DRelativeSelfAttention(nn.Module):
#     def __init__(self, num_heads:int=9,
#                   hidden_size:int=256,
#                   max_position_embeddings:int=16,
#                   dropout:float=0.1,
#                 use_attention_data=True,
#                  query_positional_score=True):
#         super().__init__()
#         self.num_attention_heads = num_heads
#         self.use_attention_data = use_attention_data
#         self.query_positional_score = query_positional_score
#         self.hidden_size = hidden_size
#         self.all_head_size = hidden_size * self.num_attention_heads
#
#
#         position_embedding_size = hidden_size
#         if self.query_positional_score:
#             position_embedding_size = hidden_size // 2
#
#
#         self.row_embeddings = nn.Embedding(2 * max_position_embeddings - 1, position_embedding_size)
#         self.col_embeddings = nn.Embedding(2 * max_position_embeddings - 1, position_embedding_size)
#
#         if not self.query_positional_score:
#             self.head_keys_row = nn.Linear(position_embedding_size, self.num_attention_heads, bias=False)
#             self.head_keys_col = nn.Linear(position_embedding_size, self.num_attention_heads, bias=False)
#
#         # need query linear transformation
#         if self.use_attention_data or self.query_positional_score:
#             self.query = nn.Linear(hidden_size, self.all_head_size)
#
#         # need key linear transformation
#         if self.use_attention_data:
#             self.key = nn.Linear(hidden_size, self.all_head_size)
#
#         self.dropout = nn.Dropout(dropout)
#         self.value = nn.Linear(self.all_head_size, hidden_size)
#
#         deltas = torch.arange(max_position_embeddings).view(1, -1) - torch.arange(max_position_embeddings).view(-1, 1)
#         # shift the delta to [0, 2 * max_position_embeddings - 1]
#         relative_indices = deltas + max_position_embeddings - 1
#
#         self.register_buffer("relative_indices", relative_indices)
#
#     def forward(self, hidden_states, head_mask=None):
#         assert len(hidden_states.shape) == 4
#         b,c,h,w = hidden_states.shape
#         #[b,c,h,w]->[b,h,w,c]
#         hidden_states = hidden_states.permute(0,2,3,1)
#
#         # -- B, W, H, num_heads, W, H
#         attention_scores = self.compute_attention_scores(hidden_states)
#         shape = attention_scores.shape
#         attention_probs = torch.nn.functional.softmax(attention_scores.view(*shape[:-2], -1),dim=-1).view(shape)
#         # attention_probs = nn.Softmax(dim=-1)(attention_scores.view(*shape[:-2], -1)).view(shape)
#         # expand batch dim if 1
#         if shape[0] != b:
#             attention_probs = attention_probs.expand(b, *shape[1:])
#
#         attention_probs = self.dropout(attention_probs)
#
#         input_values = torch.einsum('bijhkl,bkld->bijhd', attention_probs, hidden_states)
#         input_values = input_values.contiguous().view(b, h, w, -1)
#         output_value = self.value(input_values)
#
#         return output_value.permute(0,3,1,2),attention_scores
#         # if self.output_attentions:
#         #     attention_scores_per_type["attention_scores"] = attention_scores
#         #     attention_scores_per_type["attention_probs"] = attention_probs
#         #     return attention_scores_per_type, output_value
#         # else:
#         #     return output_value
#
#     def compute_attention_scores(self, hidden_states):
#         """Compute the positional attention for an image of size width x height
#         Returns: tensor of attention scores (1 or batch, width, height, num_head, width, height)
#
#         Attention scores:
#             * Position only
#                 Options: use_attention_data=False, query_positional_score=False
#                 w_q^T * r
#                 where w_q is a learned vector per head
#             * Query and positional encoding (without query key attention scores),
#                 same as q * r in (Ramachandran et al., 2019)
#                 Options: use_attention_data=False, query_positional_score=True
#                 X * W_Q * r
#             * With data
#                 same as q*k + q*r in (Ramachandran et al., 2019)
#                 Options: use_attention_data=True, query_positional_score=True
#                 X * W_Q * W_K^T * X^T + X * W_Q * r
#             * Last option use_attention_data=True, query_positional_score=False was not used
#         """
#         batch_size, height, width, hidden_dim = hidden_states.shape
#
#         # compute query data if needed
#         if self.use_attention_data or self.query_positional_score:
#             q = self.query(hidden_states)
#             q = q.view(batch_size, height, width, self.num_attention_heads, self.hidden_size)
#
#         # compute key data if needed
#         if self.use_attention_data:
#             k = self.key(hidden_states)
#             k = k.view(batch_size, height, width, self.num_attention_heads, self.hidden_size)
#
#         # Compute attention scores based on position
#         # Probably not optimal way to order computation
#         relative_indices = self.relative_indices[:width, :width].reshape(-1)
#         row_embeddings = self.row_embeddings(relative_indices)
#
#         relative_indices = self.relative_indices[:height, :height].reshape(-1)
#         col_embeddings = self.col_embeddings(relative_indices)
#
#         # keep attention scores/prob for plotting
#         # attention_scores_per_type = {}
#         sqrt_normalizer = math.sqrt(self.hidden_size)
#
#         if not self.query_positional_score:
#             # Caveat: sqrt rescaling is not used in this case
#             row_scores = self.head_keys_row(row_embeddings).view(1, width, 1, width, self.num_attention_heads)
#             col_scores = self.head_keys_col(col_embeddings).view(height, 1, height, 1, self.num_attention_heads)
#             # -- H, W, H, W, num_attention_heads
#             attention_scores = row_scores + col_scores
#             # -- H, W, num_attention_heads, H, W
#             attention_scores = attention_scores.permute(0, 1, 4, 2, 3)
#             # -- 1, H, W, num_attention_heads, H, W
#             attention_scores = attention_scores.unsqueeze(0)
#
#             # attention_scores_per_type["w_q^Tr"] = attention_scores
#
#         else:  # query_positional_score
#             # B, W, H, num_attention_heads, D // 2
#             q_row = q[:, :, :, :, :self.hidden_size // 2]
#             q_col = q[:, :, :, :, self.hidden_size // 2:]
#
#             row_scores = torch.einsum("bijhd,ikd->bijhk", q_row, row_embeddings.view(width, width, -1))
#             col_scores = torch.einsum("bijhd,jld->bijhl", q_col, col_embeddings.view(height, height, -1))
#
#             # -- B, H, W, num_attention_heads, H, W
#             attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)
#             attention_scores = attention_scores / sqrt_normalizer
#
#             # save
#             # attention_scores_per_type["q^Tr"] = attention_scores
#
#         # Compute attention scores based on data
#         if self.use_attention_data:
#             attention_content_scores = torch.einsum("bijhd,bklhd->bijhkl", q, k)
#             attention_content_scores = attention_content_scores / sqrt_normalizer
#             attention_scores = attention_scores + attention_content_scores
#
#             # # save
#             # attention_scores_per_type["q^Tk"] = attention_content_scores
#
#         # return attention_scores, attention_scores_per_type
#         return attention_scores
#
#     def get_attention_probs(self, width, height):
#         """LEGACY
#         Compute the positional attention for an image of size width x height
#         Returns: tensor of attention probabilities (width, height, num_head, width, height)
#         """
#         relative_indices = self.relative_indices[:width, :width].reshape(-1)
#         row_scores = self.head_keys_row(self.row_embeddings(relative_indices)).view(1, width, 1, width,
#                                                                                     self.num_attention_heads)
#
#         relative_indices = self.relative_indices[:height, :height].reshape(-1)
#         col_scores = self.head_keys_col(self.col_embeddings(relative_indices)).view(height, 1, height, 1,
#                                                                                     self.num_attention_heads)
#
#         # -- H, W, H, W, num_attention_heads
#         attention_scores = row_scores + col_scores
#         # -- H, W, num_attention_heads, H, W
#         attention_scores = attention_scores.permute(0, 1, 4, 2, 3)
#
#         # -- H, W, num_attention_heads, H, W
#         flatten_shape = [height, width, self.num_attention_heads, height * width]
#         unflatten_shape = [height, width, self.num_attention_heads, height, width]
#         attention_probs = nn.Softmax(dim=-1)(attention_scores.view(*flatten_shape)).view(*unflatten_shape)
#
#         return attention_probs