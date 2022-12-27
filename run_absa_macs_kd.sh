#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
mbert_path=bert-base-multilingual-cased
# xlmr_path=xlm-roberta-base

for seed in 42 52 62
    do
    for target in es nl ru fr
        do
        python main.py --tfm_type mbert \
                    --exp_type macs_kd \
                    --model_name_or_path $mbert_path \
                    --data_dir data \
                    --src_lang en \
                    --tgt_lang ${target} \
                    --do_train \
                    --do_distill \
                    --do_eval \
                    --seed ${seed} \
                    --ignore_cached_data \
                    --per_gpu_train_batch_size 16 \
                    --per_gpu_eval_batch_size 16 \
                    --learning_rate 4e-5 \
                    --tagging_schema BIEOS \
                    --overwrite_output_dir \
                    --max_steps 2000 \
                    --train_begin_saving_step 1000 \
                    --eval_begin_end 1000-2000
        done
    done