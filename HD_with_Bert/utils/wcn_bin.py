import torch
import torch.nn as nn
import torch.nn.functional as F


def length_reorder(tensor, lens):
    '''
    reorder by descending order of lens, for LSTM input
    tensor: (b, seq, d)
    lens: (b, ), list
    '''
    device = tensor.device
    lens = torch.LongTensor(lens)
    sorted_lens, sort_indices = torch.sort(lens, descending=True)
    sorted_tensor = torch.index_select(tensor, dim=0, index=sort_indices.to(device))
    sorted_lens = sorted_lens.tolist()

    return sorted_tensor, sorted_lens, sort_indices


def length_order_back(sorted_tensor, sort_indices):
    '''
    order back
    sorted_tensor: (b, seq, d')
    sort_indices: (b, )
    '''
    ori_tensor = torch.empty_like(sorted_tensor)
    ori_tensor[sort_indices] = sorted_tensor

    return ori_tensor
