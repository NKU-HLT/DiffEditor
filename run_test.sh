#!/bin/bash

ckpt_dir_name=$1
test_file_path=$2
test_out_dir=${1}_auto/${3}
test_wav_directory=$4
mode=$5

# 1 en      2 zh            3 cs
# 若果是全部测试任务，这里的test_out_dir最后请取名为zh2zh zh2en zh2cs里面的一个
# CUDA_VISIBLE_DEVICES=2 python asr.py --data_dir ./inference/$test_out_dir  --mode 3


# test_file_path = 'test_dir/zhtest.csv'
# test_wav_directory = 'test_dir/audio'
# dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
# acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
# output_directory = 'test_dir/audio/mfa_out'
# test_out_dir = hparams['work_dir'].split('/')[1]+"_auto/zhtest_knn0.1_en"



python inference/tts/spec_denoiser.py --exp_name $ckpt_dir_name   --test_file_path $test_file_path --test_out_dir $test_out_dir  --test_wav_directory  $test_wav_directory


# 1 en      2 zh            3 cs
# 若果是全部测试任务，这里的test_out_dir最后请取名为zh2zh zh2en zh2cs里面的一个
# CUDA_VISIBLE_DEVICES=7 python asr.py --data_dir ./inference/$test_out_dir  --mode $mode


# bash asr_wer_compute.sh  ${ckpt_dir_name}_auto   $mode $3

# python asv_wavlm.py --data_dir ${ckpt_dir_name}_auto  --mode $mode  --text_name $3
