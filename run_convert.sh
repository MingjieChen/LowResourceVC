#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/
root=/share/mini1/res/t/vc/studio/timap-en/vctk
#exp=$root/exp/vc-gan/0722stgan3_0

num_spks=109
exp=$root/exp/vc-gan/1006stgan1
#mc_dir=$root/dump/0825mc_109spk_22050
#mc_dir=$root/dump/0721mc_10spk_22050/
mc_dir=$root/dump/0915mc_109spk_22050_few_shot20
#mc_dir=$root/dump/0921mc_109spk_22050_few_shot5
iters=200000
source_speaker=p232
speaker_encoder_model=SPEncoder
#generator_model=LSGen
#generator_model=AdaGenSplit
generator_model=Gen
#res_block=ResidualBlockSplit
res_block=Style2ResidualBlock1DBeta
gen_speaker_emb=false
convert_all_spks=true
if [ "$gen_speaker_emb" = true ]  && [ $generator_model = AdaGen ]
then
    $PYTHON $root/vc_gan/speaker_embed.py \
                        --plot \
                        --save \
                        --model_save_dir ${exp}/ckpt/ \
                        --resume_iters $iters  \
                        --mc_test_dir $mc_dir/test/ \
                        --plot_output_dir $exp/plot/ \
                        --save_output_dir $exp/spk_emb\
                        --num_speakers $num_spks \
                        --spenc_model $speaker_encoder_model\
                        --speaker_path $mc_dir/speaker_used.json \
                        --num_workers 20 \
                        --spk_cls
fi

if [ "$convert_all_spks" = true ]
then
    $PYTHON $root/vc_gan/convert.py --wav_dir $root/resmp_wav22050 \
                        --model_save_dir ${exp}/ckpt/ \
                        --resume_iters $iters \
                        --train_data_dir $mc_dir/train/ \
                        --test_data_dir $mc_dir/test/ \
                        --convert_dir $exp/converted_samples_loudnorm/ \
                        --num_speakers $num_spks \
                        --generator $generator_model\
                        --spenc $speaker_encoder_model\
                        --res_block $res_block \
                        --sample_rate 22050 \
                        --speaker_path $mc_dir/speaker_used.json \
                        --pair_list_path $exp/pair_list.txt\
                        --num_converted_wavs 2 \
                        #--use_ema

else
    for src in p232 p229 p262 p272 p293 p251 p360 p361 p292 p248
    do
        for trg in p232 p229 p262 p272 p293 p251 p360 p361 p292 p248
        do
            if [ $src != $trg ]
            then
                $PYTHON $root/vc_gan/convert.py \
                                    --wav_dir $root/resmp_wav22050 \
                                    --model_save_dir ${exp}/ckpt/ \
                                    --resume_iters $iters \
                                    --train_data_dir $mc_dir/train/ \
                                    --test_data_dir $mc_dir/test/ \
                                    --convert_dir $exp/converted_samples_loudnorm1/ \
                                    --num_speakers $num_spks \
                                    --generator $generator_model\
                                    --res_block $res_block \
                                    --spenc $speaker_encoder_model\
                                    --sample_rate 22050 \
                                    --speaker_path $mc_dir/speaker_used.json \
                                    --pair_list_path $exp/pair_list.txt\
                                    --num_converted_wavs 10 \
                                    --src_spk $src \
                                    --trg_spk $trg \
                                    --use_loudnorm \
                                    --use_ema 
            fi
        done
    done
fi
