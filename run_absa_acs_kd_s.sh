#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
mbert_path=bert-base-multilingual-cased

python main.py --tfm_type mbert \
            --exp_type acs_kd_s \
            --model_name_or_path $mbert_path \
            --data_dir ./data \
            --src_lang en \
            --tgt_lang fr \
            --do_train \
            --do_distill \
            --do_eval \
            --task absa \
            --ignore_cached_data \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 16 \
            --learning_rate 4e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 1000 \
            --train_begin_saving_step 500 \
            --eval_begin_end 500-1000