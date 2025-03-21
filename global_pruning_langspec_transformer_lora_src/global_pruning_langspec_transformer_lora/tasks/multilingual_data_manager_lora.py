import logging
import torch
from fairseq.data import RawLabelDataset
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager
from global_pruning_langspec_transformer_lora.data import LanguagePairDatasetLora


logger = logging.getLogger(__name__)

SRC_DICT_NAME = 'src'
TGT_DICT_NAME = 'tgt'


class MultilingualDatasetManagerLora(MultilingualDatasetManager):

    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return MultilingualDatasetManagerLora(
            args, lang_pairs, langs, dicts, sampling_method
        )

    def load_langpair_dataset(
        self,
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        src_dataset_transform_func=lambda dataset: dataset,
        tgt_dataset_transform_func=lambda dataset: dataset,
        src_lang_id=None,
        tgt_lang_id=None,
        langpairs_sharing_datasets=None,
    ):
        norm_direction = "-".join(sorted([src, tgt]))
        if langpairs_sharing_datasets is not None:
            src_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src), "NotInCache"
            )
            tgt_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, tgt), "NotInCache"
            )
            align_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src, tgt), "NotInCache"
            )

        # a hack: any one is not in cache, we need to reload them
        if (
            langpairs_sharing_datasets is None
            or src_dataset == "NotInCache"
            or tgt_dataset == "NotInCache"
            or align_dataset == "NotInCache"
            or split != getattr(self.args, "train_subset", None)
        ):
            # source and target datasets can be reused in reversed directions to save memory
            # reversed directions of valid and test data will not share source and target datasets
            src_dataset, tgt_dataset, align_dataset = self.load_lang_dataset(
                data_path,
                split,
                src,
                src_dict,
                tgt,
                tgt_dict,
                combine,
                dataset_impl,
                upsample_primary,
                max_source_positions=max_source_positions,
                prepend_bos=prepend_bos,
                load_alignments=load_alignments,
                truncate_source=truncate_source,
            )
            src_dataset = src_dataset_transform_func(src_dataset)
            tgt_dataset = tgt_dataset_transform_func(tgt_dataset)
            if langpairs_sharing_datasets is not None:
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src)
                ] = src_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, tgt)
                ] = tgt_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src, tgt)
                ] = align_dataset
                if align_dataset is None:
                    # no align data so flag the reverse direction as well in sharing
                    langpairs_sharing_datasets[
                        (data_path, split, norm_direction, tgt, src)
                    ] = align_dataset
        else:
            logger.info(
                f"Reusing source and target datasets of [{split}] {tgt}-{src} for reversed direction: "
                f"[{split}] {src}-{tgt}: src length={len(src_dataset)}; tgt length={len(tgt_dataset)}"
            )

        # ["en", "de", "nl", "es", "pt", "ro", "it", "ru", "pl", "cs", "bg", "sr", "ja", "zh-cn"]
        language_dict = {'en': 0, 'fr': 1, 'it': 2, 'de': 3, 'nl': 4, 'zh':5, 'ja':6, 'ko':7, 'oc':8, 'or':9, 'sd':10, 'wo':11}

        single_src_direction = language_dict[src]
        single_tgt_direction = language_dict[tgt]
        src_direction = torch.LongTensor([single_src_direction for _ in range(len(src_dataset))])
        src_direction = RawLabelDataset(src_direction)
        tgt_direction = torch.LongTensor([single_tgt_direction for _ in range(len(tgt_dataset))])
        tgt_direction = RawLabelDataset(tgt_direction)

        return LanguagePairDatasetLora(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset.sizes if tgt_dataset is not None else None,
            tgt_dict,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )