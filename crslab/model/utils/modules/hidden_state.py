import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

def MeanHiddenStateFunc(hidden_state, mask):
    # hidden_state = (bs, seq_len, dim)
    # mask = (bs, seq_len)

    mean_hidden_states = []
    for sample_hidden_state, sample_mask in zip(hidden_state, mask):
        # sample_hidden_state = (seq_len, dim)
        # sample_mask = (seq_len)
        sample_mean_hidden_state = mask_sample_hidden_state_func(sample_hidden_state, sample_mask)  # (1, dim)
        mean_hidden_states.append(sample_mean_hidden_state)

    avg_hidden_state = torch.cat(mean_hidden_states, dim=0)  # (bs, dim)

    return avg_hidden_state

def MeanHiddenStateFunc2(hidden_state, mask_bs_n_seqlen):
    # hidden_state = (bs, seq_len, dim)
    # mask_bs_n_seqlen = # [bs, n]*(seq_len)

    masked_hidden_states = []
    for sample_hidden_state, masks in zip(hidden_state, mask_bs_n_seqlen):
        # sample_hidden_state = (seq_len, dim)
        # masks = [n]*(seq_len)
        sample_mean_hidden_states = []  # [n]*(1, dim)
        for sample_mask in masks:
            # sample_mask = (seq_len)
            sample_mean_hidden_state = mask_sample_hidden_state_func(sample_hidden_state, sample_mask) # (1, dim)
            sample_mean_hidden_states.append(sample_mean_hidden_state)
        masked_hidden_states.append(sample_mean_hidden_states)

    return masked_hidden_states # [bs, n], (1, dim)
    
def mask_sample_hidden_state_func(sample_hidden_state, sample_mask):
    # sample_hidden_state = (seq_len, dim), sample_mask = (seq_len)

    sample_mask = sample_mask.unsqueeze(1)  # (seq_len, 1)
    masked_hidden_state = sample_hidden_state * sample_mask  # (seq_len, dim)

    nb_token = torch.sum(sample_mask)
    sum_hidden_state = torch.sum(masked_hidden_state, dim=0)
    sample_mean_hidden_state = sum_hidden_state if nb_token == 0 else sum_hidden_state / nb_token

    sample_mean_hidden_state = sample_mean_hidden_state.unsqueeze(0) # (1, dim)

    return sample_mean_hidden_state # (1, dim)