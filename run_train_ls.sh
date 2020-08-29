#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/

root=/share/mini1/res/t/vc/studio/timap-en/vctk/
wav_dir=$root/resmp_wav22050/
#mc_dir=$root/dump/0721mc_10spk_22050/
mc_dir=$root/dump/0825mc_109spk_22050

exp_root=/share/mini1/res/t/vc/studio/timap-en/vctk/exp/vc-gan/
exp_name=0827stgan2_0/
exp=$exp_root/$exp_name


main_script=$root/vc-gan/main_st2ls.py

 $PYTHON $main_script \
                    --wav_dir $wav_dir \
                    --model_save_dir ${exp}/ckpt/ \
                    --sample_step 10000 \
                    --model_save_step 10000\
                    --log_dir ${exp}/tb/\
                    --num_speakers 109 \
                    --train_data_dir $mc_dir/train \
                    --test_data_dir $mc_dir/test \
                    --sample_dir $exp/samples/ \
                    --num_workers 4 \
                    --lambda_id 5.0 \
                    --drop_id_step 10000 \
                    --min_length 256 \
                    --test_src_spk p229 \
                    --test_trg_spk p232 \
                    --sampling_rate 22050 \
                    --speaker_path $mc_dir/speaker_used.json \
                    --batch_size 8 \
                    --few_shot 20\
                    --resume_iters 130000
                   
