# Language-specific LoRA

Codes and scripts of our paper [Exploring Intrinsic Language-specific Subspaces in Fine-tuning Multilingual Neural Machine Translation](https://openreview.net/forum?id=xXN16BtjnL#discussion).

## Install fairseq & apex

```bash
conda install cudatoolkit-dev=11.7 -c conda-forge

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

git clone https://github.com/NVIDIA/apex
cd apex
git checkout 6943fd26e04c59327de32592cf5af68be8f5c44e
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## Training

```bash
cd $fairseq

CUDA_VISIBLE_DEVICES=0 fairseq-train $flores101_lang12_bin \
    --update-freq 4 \
    --seed 2 --ddp-backend=no_c10d --amp \
    --user-dir /cl/work5/zhe-ca/myFairseq/fairseq/models/langspec_transformer_lora_src/langspec_transformer_lora \
    --finetune-from-model $m2m100_615m_pretrain_path \
    --arch langspec_m2m100_lora_base \
    --task translation_multi_simple_epoch_lora \
    --sampling-method "temperature" --sampling-temperature 5.0 \
    --encoder-langtok src --decoder-langtok \
    --langs $m2m100_615m_lang12_langs \
    --lang-pairs $m2m100_615m_lang12_all_langs_pairs \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-08 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 3e-03 \
    --warmup-updates 1 \
    --max-tokens 4000 \
    --max-epoch 15 \
    --required-batch-size-multiple 1 \
    --keep-last-epochs 15 \
    --log-interval 2 \
    --log-format simple \
    --save-dir /cl/work5/zhe-ca/TransformerLoRA/checkpoints/pruning_langspec_lora/pruning_ex6_3090_test \
    --decoder-layerdrop 0 \
    --decoder-layers 12 \
    --decoder-normalize-before \
    --dropout 0.1 \
    --encoder-layerdrop 0 \
    --encoder-layers 12 \
    --encoder-normalize-before \
    --label-smoothing 0.2 \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --add-lora True \
    --freeze-original-parameters True \
    --encoder-lora-layer "0_1_2_3_4_5_6_7_8_9_10_11" \
    --encoder-lora-position "q_k_v_fc1_fc2" \
    --decoder-lora-layer "0_1_2_3_4_5_6_7_8_9_10_11" \
    --decoder-lora-position "q_k_v_crossq_crossk_crossv_fc1_fc2" \
    --dim 1024 \
    --high-rank 2 \
    --med-rank 2 \
    --low-rank 8 \
    --encoder-activation-direction "src_src_src_src_src_src_src_src_src_tgt_tgt_tgt" \
    --decoder-activation-direction "tgt_tgt_tgt_tgt_tgt_tgt_tgt_tgt_tgt_tgt_tgt_tgt" \
    --language-num 12 \
    --pruning True \
    --pruning-t1 2 \
    --pruning-t2 10 \
    --target-pruning-ratio 0.9
```



## Evaluation

```bash
checkpoint_path=/cl/work5/zhe-ca/TransformerLoRA/checkpoints/pruning_langspec_lora/pruning_ex6_3090/checkpoint15.pt

cd $fairseq

out_dir=/cl/work5/zhe-ca/TransformerLoRA/scripts/pruning_langspec_transformer_lora/results/pruning_ex6/pruning_ex6_e15
mkdir -p $out_dir

langs=$(echo $m2m100_615m_lang12_langs | tr ',' ' ')

for src in $langs ; do
    for tgt in $langs ; do
        if [[ $src != $tgt ]] ; then 
            CUDA_VISIBLE_DEVICES=0 fairseq-generate $flores101_lang12_bin \
                --user-dir /cl/work5/zhe-ca/myFairseq/fairseq/models/langspec_transformer_lora_src/langspec_transformer_lora \
                --path $checkpoint_path \
                --fixed-dictionary $m2m100_615m_dict \
                --required-batch-size-multiple 1 \
                --langs $m2m100_615m_lang12_langs \
                --lang-pairs $m2m100_615m_lang12_all_langs_pairs \
                -s $src -t $tgt \
                --beam 5 \
                --remove-bpe 'sentencepiece' \
                --task translation_multi_simple_epoch_lora \
                --decoder-langtok \
                --encoder-langtok src \
                --gen-subset test \
                --fp16 \
                --dataset-impl mmap \
                --distributed-world-size 1 \
                --distributed-no-spawn > $out_dir/$src-$tgt.raw.txt
            
            cat $out_dir/$src-$tgt.raw.txt  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $out_dir/$src-$tgt.spbleu.h

            echo $src-$tgt >> $out_dir/all_direction.spbleu
            sacrebleu $flores101_lang12_dir/mono/test.$tgt < $out_dir/$src-$tgt.spbleu.h --tokenize spm -w 2 >> $out_dir/all_direction.spbleu
        fi
    done
done
```


