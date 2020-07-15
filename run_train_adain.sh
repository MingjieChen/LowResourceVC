#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/

#exp=exp/0705stada_doub0/
exp=exp/0715stadain1/
main_script=main_stgan_adain.py
#main_script=main_stadain_doubdis.py
$PYTHON $main_script \
                    --device 1\
                    --wav_dir resmp_wav22050/ \
                    --model_save_dir ${exp}/ckpt/ \
                    --sample_step 10000 \
                    --model_save_step 10000\
                    --log_dir ${exp}/tb/\
                    --num_speakers 4 \
                    --train_data_dir dump/mc/train/ \
                    --test_data_dir dump/mc/eval/ \
                    --sample_dir ./samples/$exp/ \
                    --num_workers 8 \
                    --n_critic 1\
                    --g_lr 0.0002\
                    --lambda_id 5.0  \
                    --lambda_rec 10.0 \
                    --lambda_adv 1.0 \
                    --lambda_spid 1.0 \
                    --min_length 256 \
                    --test_src_spk VCC2SF1 \
                    --test_trg_spk VCC2SM1 \
                    --sampling_rate 22050 \
                    --speaker_path ./speaker_used.json \
                    --batch_size 8 \
                    #--resume_iters 110000
                   
