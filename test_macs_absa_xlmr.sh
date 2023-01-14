#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
mbert_path=bert-base-multilingual-cased
# xlmr_path=xlm-roberta-base
#

for xlmr_path in xlm-align-base-42
    do
    for seed in 42
        do
        for target in ru
            do
            mkdir -p results_log${xlmr_path}
            mkdir -p outputDIR${xlmr_path}
            python main.py --tfm_type xlmr \
                        --exp_type macs_kd \
                        --model_name_or_path $xlmr_path \
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
                        --max_steps 2578 \
                        --train_begin_saving_step 200 \
                        --results_log results_log${xlmr_path} \
                        --outputDIR outputDIR${xlmr_path} \
                        --eval_begin_end 1500-2578
            done
        done
    done
