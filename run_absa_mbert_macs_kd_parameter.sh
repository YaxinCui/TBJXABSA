#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
mbert_path=bert-base-multilingual-cased
# model_path=xlm-roberta-base
#

for model_path in mbert-SemEvalAndAmazon10000CS5 bert-base-multilingual-cased
    do
    for seed in 42 52 62
        do
        for target in ru es fr nl
            do
            for max_steps in 2
                do
                mkdir -p results_logBIEOS/${model_path}-${max_steps}
                mkdir -p outputDIRBIEOS/${model_path}-${max_steps}
                nohup python main.py --tfm_type mbert \
                            --exp_type macs_kd \
                            --model_name_or_path $model_path \
                            --data_dir data \
                            --src_lang en \
                            --tgt_lang ${target} \
                            --task absa \
                            --do_train \
                            --do_distill \
                            --do_eval \
                            --seed ${seed} \
                            --ignore_cached_data \
                            --per_gpu_train_batch_size 16 \
                            --per_gpu_eval_batch_size 16 \
                            --learning_rate 5e-5 \
                            --tagging_schema BIEOS \
                            --overwrite_output_dir \
                            --max_steps $(expr 1289 \* $max_steps - 1) \
                            --save_steps 100 \
                            --train_begin_saving_step $(expr 1289 \* $max_steps / 2) \
                            --results_log results_logBIEOS/${model_path}-${max_steps} \
                            --outputDIR outputDIRBIEOS/${model_path}-${max_steps} \
                            --eval_begin_end $(expr 1289 \* $max_steps / 2)-$(expr 1289 \* ${max_steps} - 1) > outputDIRBIEOS/${model_path}-${max_steps}/output.text 2>&1
                done
            done
        done
    done
