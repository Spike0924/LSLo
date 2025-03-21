import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

res_table = {
    'en': 'high',
    'fr': 'high',
    'it': 'high',
    'de': 'high',
    'nl': 'med',
    'zh': 'med',
    'ja': 'med',
    'ko': 'med',
    'oc': 'low',
    'or': 'low',
    'sd': 'low',
    'wo': 'low',
}
language_dict = {'en': 0, 'fr': 1, 'it': 2, 'de': 3, 'nl': 4, 'zh':5, 'ja':6, 'ko':7, 'oc':8, 'or':9, 'sd':10, 'wo':11}

lang2id = {
    0: 'en',
    1: 'fr',
    2: 'it',
    3: 'de',
    4: 'nl',
    5: 'zh',
    6: 'ja',
    7: 'ko',
    8: 'oc',
    9: 'or',
    10: 'sd',
    11: 'wo'
}

class Lora(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        # self.lora_a = nn.Linear(dim, rank)
        # self.lora_b = nn.Linear(rank, dim)
        # nn.init.zeros_(self.lora_b.weight)
        # nn.init.zeros_(self.lora_b.bias)
    
        self.lora_a = nn.Parameter(torch.zeros(dim, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, dim))
        self.dim = dim
        self.rank = rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        # return self.lora_b(self.lora_a(x))
        return x @ self.lora_a @ self.lora_b

    def extra_repr(self) -> str:
        return f"LoRA setting: dim={self.dim}, rank={self.rank}"

class LangSpecLora(nn.Module):
    def __init__(self, language_num: int, dim: int, high_rank: int, med_rank: int, low_rank: int, activation_direction: str):
        super().__init__()
        self.language_num = language_num
        self.dim = dim
        self.high_rank = high_rank
        self.med_rank = med_rank
        self.low_rank = low_rank
        self.lang_spec_lora = nn.ModuleList([])
        self.rank_list = []
        self.activation_direction = activation_direction
        for lid in range(self.language_num):
            res = res_table[lang2id[lid]]
            if res == 'high':
                self.rank_list.append(self.high_rank)
            elif res == 'low':
                self.rank_list.append(self.low_rank)
            else:
                self.rank_list.append(self.med_rank)

        self.lang_spec_lora.extend([
            Lora(dim=self.dim, rank=self.rank_list[lid],) for lid in range(self.language_num)
        ])
    
    def forward(self, x, src_direction, tgt_direction):

        if self.activation_direction == 'src':
            direction = src_direction
        elif self.activation_direction == 'tgt':
            direction = tgt_direction
        
        output = torch.zeros_like(x)
        for i in range(self.language_num):
            mask = (direction == i)
            if mask.sum() == 0:
                continue
            selected_x = x[:, mask, :]
            tmp = self.lang_spec_lora[i](selected_x)
            if tmp.dtype != output.dtype:
                tmp = tmp.to(dtype=output.dtype)
            output[:, mask, :] = tmp
        return output
        
    def extra_repr(self) -> str:
        return f"activation_direction:{self.activation_direction}"

if __name__=="__main__":
    pass
