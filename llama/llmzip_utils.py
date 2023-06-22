import torch
import numpy as np

import numpy as np

def gen_rank(probs,next_token):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True,stable=True) 
    rank_list = []
    if next_token.shape[0]>1:
        for i in range(next_token.shape[0]):
            rank_list += [torch.where(probs_idx[i:i+1,:] == next_token[i])[-1]]
        rank = torch.squeeze(torch.stack(rank_list))
    else:
        rank = torch.where(probs_idx == next_token)[-1]
    return rank

def gen_next_token(probs,rank):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True,stable=True)
    next_token = torch.gather(probs_idx, -1, rank)
    return next_token

def read_bitstream(bitin):
    temp_list = []
    while True:
        temp = bitin.read()
        if temp == -1:
            break
        temp_list += [temp]
    temp_arr = np.array(temp_list)
    final_ind = (np.where(temp_arr==1)[0][-1]).astype(int)
    final_arr = temp_arr[:final_ind+1]
    
    return final_arr

def get_str_array(array):
    array_used = array.reshape(-1)
    str_out = str()
    for i in range(array_used.size):
        str_out +=str(array_used[i])+" "
    return str_out



