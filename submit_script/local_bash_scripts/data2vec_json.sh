#!/bin/bash
# export PYTHONPATH=~/github/fairseq:$PYTHONPATH
cd ~/github/fairseq

model_path=~/model/data2vec_models/data2vec_debug
mkdir -p ${model_path}

# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir /home/v-ziyangma/github/fairseq/examples/data2vec/config/audio/pretraining \
--config-name base_librispeech  \
checkpoint.save_dir=${model_path}   