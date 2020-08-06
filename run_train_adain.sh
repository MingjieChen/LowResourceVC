#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/

#exp=exp/0705stada_doub0/
root=/share/mini1/res/t/vc/studio/timap-en/vctk/
wav_dir=$root/resmp_wav22050/
mc_dir=$root/dump/0721mc_10spk_22050/

exp_root=/share/mini1/res/t/vc/studio/timap-en/vctk/exp/vc-gan/
#exp_name=0731stgan_gse_0/
exp_name=0806stgan3_0/
#exp_name=0801stgan_gse_1/
exp=$exp_root/$exp_name


#main_script=$root/vc-gan/main_stgan_adain.py
#main_script=$root/vc-gan/main_stgan_adain_gse.py
main_script=$root/vc-gan/main_stgan_adain_r1.py
#exp=exp/0715stadain1/
#main_script=main_stgan_adain.py
#main_script=main_stadain_doubdis.py
$PYTHON $main_script \
                    --device 1\
                    --wav_dir $wav_dir \
                    --model_save_dir ${exp}/ckpt/ \
                    --sample_step 10000 \
                    --model_save_step 10000\
                    --log_dir ${exp}/tb/\
                    --num_speakers 10 \
                    --train_data_dir $mc_dir/train \
                    --test_data_dir $mc_dir/test \
                    --sample_dir $exp/samples/ \
                    --num_workers 8 \
                    --n_critic 1\
                    --g_lr 0.0001\
                    --lambda_id 2.0  \
                    --lambda_gp 1.0 \
                    --lambda_rec 4.0 \
                    --lambda_adv 1.0 \
                    --lambda_spid 1.0 \
                    --min_length 256 \
                    --test_src_spk p272 \
                    --test_trg_spk p262 \
                    --sampling_rate 22050 \
                    --speaker_path $mc_dir/speaker_used.json \
                    --batch_size 8 \
                    #--resume_iters 120000
                   
