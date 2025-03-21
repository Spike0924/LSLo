from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from global_pruning_langspec_transformer_lora.tasks import MultilingualDatasetManagerLora
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.utils import FileContentsAction
from distutils.util import strtobool
import torch.nn.utils.prune as prune

@register_task("translation_multi_simple_epoch_lora")
class TranslationMultiSimpleEpochTaskLora(TranslationMultiSimpleEpochTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        parser.add_argument('--global-pruning', type=strtobool, default=False, help="pruning?")
        parser.add_argument('--cross-language', type=strtobool, default=False, help="Whether pruning happen cross-language")
        # parser.add_argument('--target-pruning-ratio', type=float, default=0, help="target pruning ratio")
        parser.add_argument('--pruning-t1', type=int, default=1, help="the epoch start pruning")
        parser.add_argument('--pruning-t2', type=int, default=1, help="the epoch stop pruning")
        parser.add_argument('--target-pruning-ratio', type=float, default=0, help="pruning ratio each epoch")

        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManagerLora.add_args(parser)

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.data_manager = MultilingualDatasetManagerLora.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )
        # self.target_pruning_ratio = args.target_pruning_ratio
        self.global_pruning = args.global_pruning
        self.pruning_t1 = args.pruning_t1
        self.pruning_t2 = args.pruning_t2
        self.target_pruning_ratio = args.target_pruning_ratio
        self.language_num = args.language_num
        self.cross_language = args.cross_language

    def begin_epoch(self, epoch, model):

        # print(self.target_pruning_ratio)
        if self.global_pruning:
            curr_pruning_ratio = None
            if self.pruning_t1 <= epoch and epoch <= (self.pruning_t2+self.pruning_t1):
                print(f"start pruning at epoch {epoch}")
                curr_pruning_ratio = self.target_pruning_ratio - self.target_pruning_ratio * (( 1 - (epoch - self.pruning_t1)/self.pruning_t2 ) ** 3)
            elif epoch > (self.pruning_t1 + self.pruning_t2):
                curr_pruning_ratio = self.target_pruning_ratio
            if curr_pruning_ratio != None:
                print(f"pruning ratio at epoch {epoch} is {curr_pruning_ratio}")
                if self.cross_language == False: # Language-Specific Pruning
                    for lang_id in range(self.language_num):
                        parameter_to_prune = list()
                        for layer_id in range(12):
                            parameter_to_prune  = parameter_to_prune + [
                                # (model.encoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_b'),

                                # (model.decoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].crossq_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].crossq_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].crossk_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].crossk_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].crossv_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].crossv_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_b'),
                            ]
                        prune.global_unstructured(tuple(parameter_to_prune), pruning_method=prune.L1Unstructured,amount=curr_pruning_ratio)
                    print('Pruning finished')
                elif self.cross_language == True: # Cross-Language Pruning
                    for layer_id in range(12):
                        encoder_parameter_to_prune = list()
                        decoder_parameter_to_prune = list()
                        for lang_id in range(self.language_num):
                            encoder_parameter_to_prune = encoder_parameter_to_prune + [
                                # (model.encoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.encoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.encoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_b'),
                            ]
                            decoder_parameter_to_prune = decoder_parameter_to_prune + [
                                # (model.decoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].crossq_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].crossq_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].crossk_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].crossk_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].crossv_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].crossv_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], 'lora_b'),
                                # (model.decoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_a'),
                                (model.decoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], 'lora_b'),
                            ]
                        
                        prune.global_unstructured(tuple(encoder_parameter_to_prune), pruning_method=prune.L1Unstructured, amount=curr_pruning_ratio)
                        prune.global_unstructured(tuple(decoder_parameter_to_prune), pruning_method=prune.L1Unstructured, amount=curr_pruning_ratio)
                        
                    print('Pruning finished')
        
    def begin_valid_epoch(self, epoch, model):
        if self.global_pruning:
            if self.pruning_t1 <= epoch:
                print(f"start remove pruning at {epoch}")
                for layer_id in range(12):
                    for lang_id in range(self.language_num):
                        # prune.remove(model.encoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.encoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.encoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.encoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.encoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.encoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.encoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.encoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.encoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.encoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], name='lora_b')

                        # prune.remove(model.decoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].q_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.decoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].k_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.decoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].v_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.decoder.layers[layer_id].crossq_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].crossq_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.decoder.layers[layer_id].crossk_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].crossk_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.decoder.layers[layer_id].crossv_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].crossv_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.decoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].fc1_lora.lang_spec_lora[lang_id], name='lora_b')
                        # prune.remove(model.decoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], name='lora_a')
                        prune.remove(model.decoder.layers[layer_id].fc2_lora.lang_spec_lora[lang_id], name='lora_b')
                print('Remove finished')