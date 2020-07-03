# VoiceConversionGANs
GAN series for voice conversion on VCC2018 dataset

## This is a voice conversion repository including cyclegan-vc, stargan-vc, stargan-vc2 and some other variants
This work is still in progress, more GAN models will be included
 
## This work is based on repository [stargan-vc](https://github.com/liusongxiang/StarGAN-Voice-Conversion), [stargan-vc2](https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2) and [cyclegan-vc](https://github.com/leimao/Voice_Converter_CycleGAN)

### Requirements:
 1. Python3
 2. PyTorch 0.4.1
 3. Pyworld

### Models:
 1. stgan: stargan-vc1 from https://github.com/liusongxiang/StarGAN-Voice-Conversion
 2. stgan2: stargan-vc2  from https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2
 3. stgan1\_cin: stargan-vc1 + generator with conditional instance normalization + speaker classifier
 4. stgan2\_new: stargan-vc2 + patchgan discriminator + only target condition in generator and discriminator + no speaker classifier + gradient penalty
 5. stgan2\_ls: stargan-vc2 + projection discriminator (as in the paper) + source and target conditions in generator and discriminator + LSGAN adversarial loss
 6. cycgan: cyclegan-vc1

### Preprocess
> ./run\_pre.sh

Modify according to your own conda env and hyper-params

### Train:
> ./run\_train.sh

Modify according to your own conda env and hyper-params

### Convert
> ./run\_convert.sh

### Objective Evaluation
> ./run\_eval.sh

This evaluation calculate Mel Cepstral Distortion (MCD) and Modulation Spectral Distance (MSD) as in stargan-vc2 paper.

However, this script can not get the same score as the paper.
