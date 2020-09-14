#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON main.py --wav_dir dump/wav16/
root=/share/mini1/res/t/vc/studio/timap-en/vctk
#exp=$root/exp/vc-gan/0722stgan3_0

num_spks=10
exp=$root/exp/vc-gan/0913stgan3_8
#mc_dir=$root/dump/0825mc_109spk_22050
mc_dir=$root/dump/0721mc_10spk_22050/
iters=100000
source_speaker=p232
speaker_encoder_model=SPEncoder
generator_model=AdaGenSplit
gen_speaker_emb=false
convert_all_spks=false

if [ "$gen_speaker_emb" = true ]  && [ $generator_model = AdaGen ]
then
    $PYTHON $root/vc-gan/speaker_embed.py \
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
    $PYTHON $root/vc-gan/convert.py --wav_dir $root/resmp_wav22050 \
                        --model_save_dir ${exp}/ckpt/ \
                        --resume_iters $iters \
                        --train_data_dir $mc_dir/train/ \
                        --test_data_dir $mc_dir/test/ \
                        --convert_dir $exp/converted_samples/ \
                        --num_speakers $num_spks \
                        --generator $generator_model\
                        --spenc $speaker_encoder_model\
                        --sample_rate 22050 \
                        --speaker_path $mc_dir/speaker_used.json \
                        --pair_list_path $exp/pair_list.txt\
                        --num_converted_wavs 10 \
                        #--spk_cls \
                        #--use_spk_mean \
                        #--spk_mean_dir $exp/spk_emb\
else
    for trg in p229  p251 p292 p272 p293 p262 p248 p361 p360
    #for trg in p225 p228 p229 p230 p231 p233 p234 p236 p238 p239 p240
    do
        $PYTHON $root/vc-gan/convert.py \
                            --wav_dir $root/resmp_wav22050 \
                            --model_save_dir ${exp}/ckpt/ \
                            --resume_iters $iters \
                            --train_data_dir $mc_dir/train/ \
                            --test_data_dir $mc_dir/test/ \
                            --convert_dir $exp/converted_samples/ \
                            --num_speakers $num_spks \
                            --generator $generator_model\
                            --spenc $speaker_encoder_model\
                            --sample_rate 22050 \
                            --speaker_path $mc_dir/speaker_used.json \
                            --pair_list_path $exp/pair_list.txt\
                            --num_converted_wavs 10 \
                            --src_spk $source_speaker \
                            --trg_spk $trg \
                            #--spk_cls \
                            #--drop_affine \
                            #--use_spk_mean \
                            #--spk_mean_dir $exp/spk_emb\

    done
fi
