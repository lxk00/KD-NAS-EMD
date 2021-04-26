import torch
from torch.nn.functional import softmax
from math import sqrt
from torch.nn.parallel._functions import Broadcast

def replace_masked(tensor, mask, value):
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i + len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies

def mean_max(input_tensor, mask=None):
    if mask is None:
        mask = torch.ones(input_tensor.shape[:2])
    mask = mask.unsqueeze(-1).repeat(1, 1, input_tensor.shape[-1])
    max_mid, _ = replace_masked(input_tensor, mask, -1e7).max(dim=1)
    mean_mid = torch.sum(input_tensor * mask, dim=1) / torch.sum(mask, dim=1)
    out = torch.cat([max_mid, mean_mid], dim=-1)
    return out

def convert_to_attn(hidns, mask):
    if type(hidns[0]) is not tuple:
        hdim = hidns[0].shape[-1]
        attns = [torch.matmul(x, x.transpose(2, 1)) / sqrt(hdim) for x in hidns]
        mask = mask.unsqueeze(1)
        mask = (1.0 - mask) * -10000.0
        attns = [softmax(x + mask, dim=-1) for x in attns]
    else:
        hidns = [torch.stack(x, dim=1) for x in hidns]
        hdim = hidns[0][0].shape[-1]
        attns = [torch.matmul(x, x.transpose(-1, -2)) / sqrt(hdim) for x in hidns]
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask) * -10000.0
        attns = [softmax(x + mask, dim=-1) for x in attns]
    return attns


import torch.nn as nn
import torch
import math
class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(
            hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(1).unsqueeze(1)
        mask = (1.0 - mask) * -10000.0

        query_layer = self.transpose_for_scores(hidden_states)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, query_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        return attention_probs


if __name__ == 'main':
    mask = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    mask = torch.tensor([map(float, mask.splint(' '))])
    i = torch.rand(1, 128, 768)
    b = BertSelfAttention(768, 12)
    b(i, mask)