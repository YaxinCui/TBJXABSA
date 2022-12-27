#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
mbert_path=bert-base-multilingual-cased
# xlmr_path=xlm-roberta-base
#

for mbert_path in bert-base-multilingual-cased
    do
    for seed in 42
        do
        for target in es
            do
            for max_steps in 2 4 6
                do
                mkdir -p results_log${mbert_path}-${max_steps}
                mkdir -p outputDIR${mbert_path}-${max_steps}
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
                            --max_steps ${expr 1289 \* $max_steps} \
                            --save_steps 1289 \
                            --train_begin_saving_step ${expr 1289 \* $max_steps / 2} \
                            --results_log results_log${mbert_path}-${max_steps} \
                            --outputDIR outputDIR${mbert_path}-${max_steps} \
                            --eval_begin_end ${expr 1289 \* max_steps / 2}-${max_steps}
                done
            done
        done
    done