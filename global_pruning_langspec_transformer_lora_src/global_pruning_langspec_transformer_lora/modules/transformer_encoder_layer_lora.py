from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
)
from .multihead_attention_lora import MultiheadAttentionLora
from .lora import LangSpecLora

"""
Rewrite TransformerEncoderLayerBase & TransformerDecoderLayerBase
Notes of this file have been deleted for briefing.
If you need to read, please see file ../fairseq/modules/transformer_layer.py
"""
class TransformerEncoderLayerLoraBase(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.language_num = cfg.language_num
        self.dim = cfg.dim
        # self.rank = cfg.rank
        self.high_rank = cfg.high_rank
        self.med_rank = cfg.med_rank
        self.low_rank = cfg.low_rank
        self.encoder_activation_direction = list(cfg.encoder_activation_direction.split('_') if cfg.encoder_activation_direction else None)
        assert self.encoder_activation_direction != None
        self.encoder_activation_direction = self.encoder_activation_direction[idx]
        self.encoder_lora_position = list(cfg.encoder_lora_position.split('_')) if cfg.encoder_lora_position else None
        assert self.encoder_lora_position != None
        for position in self.encoder_lora_position:
            assert position in ['k', 'q', 'v', 'fc1', 'fc2']
        
        self.q_lora = None
        if 'q' in self.encoder_lora_position:
            self.q_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.encoder_activation_direction
            )
        self.k_lora = None
        if 'k' in self.encoder_lora_position:
            self.k_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.encoder_activation_direction
            )
        self.v_lora = None
        if 'v' in self.encoder_lora_position:
            self.v_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.encoder_activation_direction
            )
        self.fc1_lora = None
        if 'fc1' in self.encoder_lora_position:
            self.fc1_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.encoder_activation_direction
            )
        self.fc2_lora = None
        if 'fc2' in self.encoder_lora_position:
            self.fc2_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.encoder_activation_direction
            )
        self.start_idx = cfg.start_idx
        self.dim=cfg.dim
        self.idx = idx

    def build_lora(self, language_num, dim, high_rank, med_rank, low_rank, activation_direction):
        return LangSpecLora(
            language_num=language_num,
            dim=dim,
            high_rank=high_rank,
            med_rank=med_rank,
            low_rank=low_rank,
            activation_direction=activation_direction,
        )

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttentionLora(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        src_direction = None,
        tgt_direction = None,
    ):
        
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # we can customize attention block's params
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            q_lora=self.q_lora,
            k_lora=self.k_lora,
            v_lora=self.v_lora,
            start_idx=self.start_idx,
            dim=self.dim,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        tmp_x = self.fc1(x)
        if self.fc1_lora != None:
            tmp_x[:, :, self.start_idx:self.start_idx+self.dim] = tmp_x[:, :, self.start_idx:self.start_idx+self.dim] + self.fc1_lora(x[:,:,self.start_idx:self.start_idx+self.dim], src_direction=src_direction, tgt_direction=tgt_direction)
        x = self.activation_fn(tmp_x)
        x = self.activation_dropout_module(x)
        tmp_x = self.fc2(x)
        if self.fc2_lora != None:
            tmp_x[:, :, self.start_idx:self.start_idx+self.dim] = tmp_x[:, :, self.start_idx:self.start_idx+self.dim] + self.fc2_lora(x[:,:,self.start_idx:self.start_idx+self.dim], src_direction=src_direction, tgt_direction=tgt_direction)
        x = self.dropout_module(tmp_x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


# backward compatible with the legacy argparse format
class TransformerEncoderLayerLora(TransformerEncoderLayerLoraBase):
    def __init__(self, args, idx=None):
        super().__init__(TransformerConfig.from_namespace(args), idx)
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )