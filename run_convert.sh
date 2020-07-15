#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/

exp=exp/0712st2ls1/
$PYTHON convert.py --wav_dir resmp_wav22050/eval/ \
                    --model_save_dir ${exp}/ckpt/ \
                    --resume_iters 260000 \
                    --train_data_dir ./dump/mc/train/ \
                    --test_data_dir ./dump/mc/eval/ \
                    --convert_dir $exp/converted_samples/ \
                    --num_speakers 4 \
                    --generator LSGen\
                    --sample_rate 22050 \
                    --speaker_path ./speaker_used.json \
                    --pair_list_path $exp/pair_list.txt
                   
