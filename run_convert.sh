#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/
root=/share/mini1/res/t/vc/studio/timap-en/vctk
#exp=$root/exp/vc-gan/0722stgan3_0
exp=$root/exp/vc-gan/0729stgan_gse_0/

$PYTHON $root/vc-gan/convert.py --wav_dir $root/resmp_wav22050 \
                    --model_save_dir ${exp}/ckpt/ \
                    --resume_iters 230000 \
                    --train_data_dir $root/dump/0721mc_10spk_22050/train/ \
                    --test_data_dir $root/dump/0721mc_10spk_22050/test/ \
                    --convert_dir $exp/converted_samples/ \
                    --num_speakers 10 \
                    --generator AdaGen\
                    --sample_rate 22050 \
                    --speaker_path $root/dump/0721mc_10spk_22050/speaker_used.json \
                    --pair_list_path $exp/pair_list.txt
                   
