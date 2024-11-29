# AimTS: Augmentation-wise and Series-Image Contrastive Learning for Time Series Classification

# Datasets

The data set is stored in`./dataset/`

# Usage

Draw pre-training images

~~~shell
# monash
bash ./script/construct_image_monash.sh
# UCR
bash ./script/construct_image_UCR.sh
# UEA
bash ./script/construct_image_UEA.sh
~~~

Pre-training (select different data sets)

~~~shell
# monash
bash ./scripts/pretrain_monash.sh
# UCR
bash ./scripts/pretrain_UCR.sh
# UEA
bash ./scripts/pretrain_UEA.sh
~~~

The pre-trained model is saved in`./checkpoints/`

You can download our pre-trained parameters directly for downstream tasks.

Fine tune and test

~~~shell
bash ./script/finetune_ALL.sh
~~~

Save the result in`./res/`