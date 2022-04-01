#!/bin/bash

#pip install tensorboardX
#pip install --upgrade numpy
#pip install --upgrade mkl-service --user
#sudo chmod -R 777 /miniconda/envs/speech_transducers/lib/python3.7/site-packages
#sudo chmod -R 777 /miniconda/envs/speech_transducers/bin
#pip install --editable ./
#export MKL_SERVICE_FORCE_INTEL=8
pip list

#edit your config
config_dir=./config/data2vec/audio/pretraining
config_name=mix_base_librispeech

#edit your data
data_path=/datablob/users/v-sanych/data/librispeech/manifest/train_960
train_subset=train_960
valid_subset=valid

# edit your compute resource
distributed_world_size=16
update_freq=[1]
max_tokens=3800000

#edit your ckpt
model_path=/datablob/users/v-sanych/model/data2vec/mix_data2vec_960h
mkdir -p ${model_path}
cp -r /tmp/code ${model_path}

echo "Start training!!!"
echo -e '\n'
# pretrain
python fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
checkpoint.save_dir=${model_path}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens}

cp -r /tmp/code/outputs/ ${model_path}

echo -e '\n'
echo "finshed!"