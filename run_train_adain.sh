#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/

#exp=exp/0705stada_doub0/
root=/share/mini1/res/t/vc/studio/timap-en/vctk/
wav_dir=$root/resmp_wav22050/
mc_dir=$root/dump/0721mc_10spk_22050/
#mc_dir=$root/dump/0825mc_109spk_22050
num_spks=10
exp_root=/share/mini1/res/t/vc/studio/timap-en/vctk/exp/vc-gan/
#exp_name=0731stgan_gse_0/a
exp_name=0914stgan3_1
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
                    --num_speakers $num_spks \
                    --train_data_dir $mc_dir/train \
                    --test_data_dir $mc_dir/test \
                    --sample_dir $exp/samples/ \
                    --num_workers 8 \
                    --n_critic 1\
                    --d_lr 0.0001\
                    --g_lr 0.0002\
                    --lambda_id 0   \
                    --lambda_gp 1.0 \
                    --lambda_rec 2 \
                    --lambda_adv 1 \
                    --lambda_spid 1 \
                    --lambda_cls 0.0  \
                    --min_length 256 \
                    --test_src_spk p232 \
                    --test_trg_spk p229 \
                    --sampling_rate 22050 \
                    --speaker_path $mc_dir/speaker_used.json \
                    --generator GeneratorSplit\
                    --discriminator PatchDiscriminator \
                    --spenc SPEncoder \
                    --batch_size 8 \
                    --drop_id_step 10000\
                    #--resume_iters 100000\
                    #--few_shot 20 \
                    #--spk_cls \
                   
