from typing import Optional
import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_legacy import (
    TransformerModel
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
)
from global_pruning_langspec_transformer_lora.models.transformer_encoder_lora import TransformerEncoderLora
from global_pruning_langspec_transformer_lora.models.transformer_decoder_lora import TransformerDecoderLora
from distutils.util import strtobool

@register_model("global_pruning_langspec_transformer_lora")
class TransformerModelLora(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args

        if self.args.freeze_original_parameters:
            for name, param in self.encoder.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
            for name, param in self.decoder.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False

        print('Trainable parameters')
        print('*'*50)
        print('Encoder')
        print('*'*50)
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                print(name)
        print('*'*50)
        print('Decoder')
        print('*'*50)
        for name, param in self.decoder.named_parameters():
            if param.requires_grad:
                print(name)
        print('*'*50)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderLora(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderLora(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        """Add model-specific arguments to the parser."""
        parser.add_argument('--language-num', default=0, type=int)
        parser.add_argument('--add-lora', default=False, type=strtobool)
        parser.add_argument("--freeze-original-parameters", default=False, type=strtobool)
        parser.add_argument("--encoder-lora-layer", default='', type=str, help="0_1_2_3..._11")
        parser.add_argument("--encoder-lora-position", default='', type=str, help="q_k_v_fc1_fc2")
        parser.add_argument("--decoder-lora-layer", default='', type=str, help="0_1_2_3_..._11")
        parser.add_argument("--decoder-lora-position", default='', type=str, help="q_k_v_crossq_crossk_crossv_fc1_fc2")
        parser.add_argument("--dim", default=0, type=int)
        # parser.add_argument("--rank", default=0, type=int)
        parser.add_argument("--start-idx", default=0, type=int)
        parser.add_argument('--encoder-activation-direction', default='', type=str, help="choos the direction (src or tgt) for lora components per layer, eg. src_src_src_...._src")
        parser.add_argument('--decoder-activation-direction', default='', type=str, help="choos the direction (src or tgt) for lora components per layer, eg. src_src_src_...._src")
        parser.add_argument('--high-rank', default=0, type=int, help="support set the lora rank according to the resource level")
        parser.add_argument('--med-rank', default=0, type=int, help="for medium resource languages")
        parser.add_argument('--low-rank', default=0, type=int, help="for low resource languages")

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        save_encoder = False,
        **kwargs,
    ):
        # When using customized multilingual dataset manager, we can get src/tgt directions respected with each batch.
        # In this case, we can train/fine-tune the model with multiple language pairs parallely.
        src_direction = kwargs.get("src_direction", None),
        tgt_direction = kwargs.get("tgt_direction", None),
        src_direction = src_direction[0]
        tgt_direction = tgt_direction[0]

        # # here is the test part
        # print("enter_forward")
        # print(f"src_tokens\n{src_tokens}")
        # print(f"src_token_shape\n{src_tokens.shape}")
        # print(f"src_lens\n{src_lengths}")
        # print(f"return_all_hiddens\n{return_all_hiddens}")
        # print(f"src_direction\n{src_direction}")
        # print(f"tgt_direction\n{tgt_direction}")
        # print(f"features_only\n{features_only}")
        # print(f"alignment_layer\n{alignment_layer}")
        # print(f"alignment_heads\n{alignment_heads}")
        # print(f"prev_token\n{prev_output_tokens}")
        # exit()
        # # test end
        # print("enter forward")

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
        )

        # if save_encoder_dir is not None:
        #     print('start save encoder_out')
        #     tmp_output = [torch.mean(item, dim=0) for item in encoder_out["encoder_states"] ]
        #     torch.save(tmp_output, save_encoder_dir)
        #     print('save finished')
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        # if save_decoder_dir is not None:
        #     print('start save decoder_out')
        #     tmp_output = decoder_out[1]["decoder_sentence"]
        #     torch.save(tmp_output, save_decoder_dir)
        if save_encoder is not False:
            mask = encoder_out["encoder_padding_mask"][0] == False
            encoder_states = encoder_out["encoder_states"][1:]
            # print(len(encoder_states))
            sentences = [(item.transpose(0,1) * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1) for item in encoder_states]
            # print(len(sentences))
            decoder_out[1]["encoder_sentence"] = sentences
        return decoder_out


@register_model_architecture("global_pruning_langspec_transformer_lora", "global_pruning_langspec_transformer_lora")
def langspec_transformer_lora_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.2)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

@register_model_architecture("global_pruning_langspec_transformer_lora", "global_pruning_langspec_m2m100_lora_base")
def m2m100_lora_base(args):
    args.sampling_method = getattr(args, "sampling_method", "temperature")
    args.sampling_temperature = getattr(args, "sampling_temperature", 5.0)
    args.label_smoothing = getattr(args, "label_smoothing", 0.2)
    args.required_batch_size_multiple = getattr(args, "required_batch_size_multiple", 1)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.label_smoothing = getattr(args, "label_smoothing", 0.2)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    langspec_transformer_lora_base_architecture(args)