#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
mbert_path=bert-base-multilingual-cased-42-SemEvalAndAmazon10000CS5-LR4e-5-PoolCLS-Temp1.0-MIN2.0-CSMLM4.0-WarmTrue/
# xlmr_path=xlm-roberta-base

python main.py --tfm_type mbert \
            --exp_type acs_mtl \
            --model_name_or_path $mbert_path \
            --data_dir data \
            --src_lang en \
            --tgt_lang fr \
            --do_train \
            --do_distill \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 16 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 1300 \
            --train_begin_saving_step 800 \
            --eval_begin_end 800-1300
