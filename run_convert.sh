#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/
root=/share/mini1/res/t/vc/studio/timap-en/vctk
#exp=$root/exp/vc-gan/0722stgan3_0

exp=$root/exp/vc-gan/0825stgan3_0
iters=30000
source_speaker=p229
speaker_encoder_model=SPEncoderPool1D
generator_model=AdaGen

$PYTHON $root/vc-gan/speaker_embed.py \
                    --plot \
                    --save \
                    --model_save_dir ${exp}/ckpt/ \
                    --resume_iters $iters  \
                    --mc_test_dir $root/dump/0721mc_10spk_22050/test/ \
                    --plot_output_dir $exp/plot/ \
                    --save_output_dir $exp/spk_emb\
                    --num_speakers 10 \
                    --spenc_model $speaker_encoder_model\
                    --speaker_path $root/dump/0721mc_10spk_22050/speaker_used.json \
                    #--spk_cls

for trg in p232  p251 p292 p272 p293 p262 p248 p361 p360
do
    $PYTHON $root/vc-gan/convert.py --wav_dir $root/resmp_wav22050 \
                        --model_save_dir ${exp}/ckpt/ \
                        --resume_iters $iters \
                        --train_data_dir $root/dump/0721mc_10spk_22050/train/ \
                        --test_data_dir $root/dump/0721mc_10spk_22050/test/ \
                        --convert_dir $exp/converted_samples/ \
                        --num_speakers 10 \
                        --generator $generator_model\
                        --spenc $speaker_encoder_model\
                        --sample_rate 22050 \
                        --speaker_path $root/dump/0721mc_10spk_22050/speaker_used.json \
                        --pair_list_path $exp/pair_list.txt\
                        --use_spk_mean \
                        --spk_mean_dir $exp/spk_emb\
                        --num_converted_wavs 10 \
                        --src_spk $source_speaker \
                        --trg_spk $trg \
                        #--spk_cls

done
