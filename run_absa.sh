#!/usr/bin/env bash

mbert_path=../trained-transformers/bert-multi-cased
xlmr_path=xlm-roberta-base

python /content/drive/MyDrive/XABSA/XABSA/main.py --tfm_type xlmr \
            --exp_type acs_mtl \
            --model_name_or_path $xlmr_path \
            --data_dir /content/drive/MyDrive/XABSA/XABSA/data \
            --src_lang en \
            --tgt_lang fr \
            --do_train \
            --do_distill \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 12 \
            --learning_rate 5e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 300 \
            --train_begin_saving_step 0 \
            --eval_begin_end 0-200 \
            --outputDIR  /content/outputs \
            --results_log  /content/results_log \
