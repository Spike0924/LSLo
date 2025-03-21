# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from .lora import LangSpecLora
from .multihead_attention_lora import MultiheadAttentionLora

"""
Rewrite TransformerEncoderLayerBase & TransformerDecoderLayerBase
Notes of this file have been deleted for briefing.
If you need to read, please see file ../fairseq/modules/transformer_layer.py

Attention!
Currently, there is not updated compared to TransformerDecoderLayer & TransformerDecoderLayerBase.
You can customize it as same with the logic of customizing TransformerEncoderLayer.
"""
class TransformerDecoderLayerLoraBase(nn.Module):
    def __init__(
        self, cfg, idx, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False
        
        self.language_num = cfg.language_num
        self.dim = cfg.dim
        # self.rank = cfg.rank
        self.high_rank = cfg.high_rank
        self.med_rank = cfg.med_rank
        self.low_rank = cfg.low_rank
        self.decoder_activation_direction = list(cfg.decoder_activation_direction.split('_') if cfg.decoder_activation_direction else None)
        assert self.decoder_activation_direction != None
        self.decoder_activation_direction = self.decoder_activation_direction[idx]
        self.decoder_lora_position = list(cfg.decoder_lora_position.split('_')) if cfg.decoder_lora_position else None
        assert self.decoder_lora_position != None
        for position in self.decoder_lora_position:
            assert position in ['q', 'k', 'v', 'crossq', 'crossk', 'crossv', 'fc1', 'fc2']
        
        self.q_lora = None
        if 'q' in self.decoder_lora_position:
            self.q_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )
        self.k_lora = None
        if 'k' in self.decoder_lora_position:
            self.k_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )
        self.v_lora = None
        if 'v' in self.decoder_lora_position:
            self.v_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )
        self.crossq_lora = None
        if 'crossq' in self.decoder_lora_position:
            self.crossq_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )
        self.crossk_lora = None
        if 'crossk' in self.decoder_lora_position:
            self.crossk_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )
        self.crossv_lora = None
        if 'crossv' in self.decoder_lora_position:
            self.crossv_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )
        self.fc1_lora = None
        if 'fc1' in self.decoder_lora_position:
            self.fc1_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )
        self.fc2_lora = None
        if 'fc2' in self.decoder_lora_position:
            self.fc2_lora = self.build_lora(
                language_num=self.language_num,
                dim=self.dim,
                high_rank=self.high_rank,
                med_rank=self.med_rank,
                low_rank=self.low_rank,
                activation_direction=self.decoder_activation_direction,
            )

        self.start_idx = cfg.start_idx
        self.dim = cfg.dim

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
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttentionLora(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttentionLora(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        src_direction = None,
        tgt_direction = None,
    ):
        if need_head_weights:
            need_attn = True
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
            q_lora=self.q_lora,
            k_lora=self.k_lora,
            v_lora=self.v_lora,
            dim=self.dim,
            start_idx = self.start_idx,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                crossq_lora=self.crossq_lora,
                crossk_lora=self.crossk_lora,
                crossv_lora=self.crossv_lora,
                src_direction=src_direction,
                tgt_direction=tgt_direction,
                start_idx=self.start_idx,
                dim=self.dim,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
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
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class TransformerDecoderLayerLora(TransformerDecoderLayerLoraBase):
    def __init__(
        self, args, idx=None, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            idx=idx,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
        )