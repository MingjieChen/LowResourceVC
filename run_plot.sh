#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/
root=/share/mini1/res/t/vc/studio/timap-en/vctk
#exp=$root/exp/vc-gan/0722stgan3_0
exp=$root/exp/vc-gan/0917stgan3_5

$PYTHON $root/vc_gan/speaker_embed.py \
                    --plot \
                    --model_save_dir ${exp}/ckpt/ \
                    --resume_iters 120000 \
                    --mc_test_dir $root/dump/0721mc_10spk_22050/test/ \
                    --plot_output_dir $exp/plot/ \
                    --save_output_dir $exp/spk_emb\
                    --num_speakers 109 \
                    --spenc_model SPEncoder\
                    --speaker_path $root/dump/0721mc_10spk_22050/speaker_used.json \
                   
