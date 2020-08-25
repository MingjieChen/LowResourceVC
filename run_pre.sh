#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python


wav_path=/share/mini1/data/audvis/pub/vc/studio/en/vctk/v1/wav48/
#eval_wav_path=/share/mini1/res/t/vc/studio/timap-en/vcc2018/dataset/vcc2018_evaluation/
root=/share/mini1/res/t/vc/studio/timap-en/vctk/
resample_path=$root/resmp_wav22050
mc_path=$root/dump/0825mc_109spk_22050/
#script_path=/share/mini1/res/t/vc/studio/timap-en/vctk/stgan/preprocess.py

python_script=$root/vc-gan/preprocess_vctk.py


$PYTHON $python_script   \
                        --do_split \
                        --sample_rate 22050 \
                        --origin_wavpath $wav_path  \
                         --target_wavpath $resample_path \
                         --mc_dir_train $mc_path/train \
                         --mc_dir_test $mc_path/test \
                         --speaker_used_path $mc_path/speaker_used.json \
                         --num_workers 20 \
                         #--do_resample \
                         #--speaker_list  p262 p272 p229 p232 p292 p293 p360 p361 p248 p251
