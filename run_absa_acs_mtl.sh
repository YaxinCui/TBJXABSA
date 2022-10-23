#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
mbert_path=bert-base-multilingual-cased
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
            --per_gpu_train_batch_size 8 \
            --per_gpu_eval_batch_size 8 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 300 \
            --train_begin_saving_step 0 \
            --eval_begin_end 0-200 \
            --outputDIR  outputs \
            --results_log  ./results_log \