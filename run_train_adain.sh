#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/

#exp=exp/0705stada_doub0/
root=/share/mini1/res/t/vc/studio/timap-en/vctk/
wav_dir=$root/resmp_wav22050/
mc_dir=$root/dump/0721mc_10spk_22050/

exp_root=/share/mini1/res/t/vc/studio/timap-en/vctk/exp/vc-gan/
#exp_name=0731stgan_gse_0/a
exp_name=0825stgan3_1
#exp_name=0801stgan_gse_1/
#exp_name=0811stadain_map_2

exp=$exp_root/$exp_name


main_script=$root/vc-gan/main_stgan_adain.py
#main_script=$root/vc-gan/main_stgan_adain_gse.py
#main_script=$root/vc-gan/main_stgan_adain_r1.py
#main_script=$root/vc-gan/main_stadain_map.py
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
                    --d_lr 0.0001\
                    --g_lr 0.0001\
                    --lambda_id 5.0  \
                    --lambda_gp 1.0 \
                    --lambda_rec 10.0 \
                    --lambda_adv 2.0 \
                    --lambda_spid 1 \
                    --lambda_cls 1e-4 \
                    --min_length 256 \
                    --test_src_spk p229 \
                    --test_trg_spk p232 \
                    --sampling_rate 22050 \
                    --speaker_path $mc_dir/speaker_used.json \
                    --discriminator PatchDiscriminator \
                    --spenc SPEncoderPool \
                    --batch_size 8 \
                    --drop_id_step 5000\
                    #--resume_iters 120000\
                    #--spk_cls \
                   
