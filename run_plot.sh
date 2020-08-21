#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/
root=/share/mini1/res/t/vc/studio/timap-en/vctk
#exp=$root/exp/vc-gan/0722stgan3_0
exp=$root/exp/vc-gan/0818stgan3_2

$PYTHON $root/vc-gan/speaker_embed.py \
                    --save \
                    --model_save_dir ${exp}/ckpt/ \
                    --resume_iters 20000 \
                    --mc_test_dir $root/dump/0721mc_10spk_22050/test/ \
                    --output_dir $exp/spk_emb/ \
                    --num_speakers 10 \
                    --spenc_model SPEncoder\
                    --speaker_path $root/dump/0721mc_10spk_22050/speaker_used.json \
                   
