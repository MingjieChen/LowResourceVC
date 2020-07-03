#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python


train_wav_path=/share/mini1/res/t/vc/studio/timap-en/vcc2018/dataset/vcc2018_training/
eval_wav_path=/share/mini1/res/t/vc/studio/timap-en/vcc2018/dataset/vcc2018_evaluation/
resample_path=./resmp_wav22050
#script_path=/share/mini1/res/t/vc/studio/timap-en/vctk/stgan/preprocess.py



$PYTHON preprocess.py   --sample_rate 22050 --origin_train_wavpath $train_wav_path --origin_eval_wavpath $eval_wav_path \
                         --target_train_wavpath $resample_path/train/ --target_eval_wavpath $resample_path/eval/ \
                         --mc_dir_train dump/mc/train --mc_dir_test dump/mc/eval \
                         --num_workers 20 --speaker_list VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2 
