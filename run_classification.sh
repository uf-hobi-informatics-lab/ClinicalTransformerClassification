#!/bin/bash

# example for how to run prediction on batch data
# see ./src/data_processing/preprocessing-batch.ipynb for example on how to generate batch data for prediction
# we assume the batch data is in ./data/batch_data
# you should expect there are several directories named as batch_* with a test.tsv file in each of them

export CUDA_VISIBLE_DEVICES=1
data_dir= /home/chenaokun1990/datasets/n2c2_2022_cls
nmd=./new_modelzw
pof=./gatortron_syn_n2c2_pred.txt
log=./log_gatortron_n2c2.txt

# NOTE: we have more options available, you can check our wiki for more information
python ./src/relation_extraction.py \
		--model_type gatortron \
		--data_format_mode 0 \
		--classification_scheme 1 \
		--pretrained_model /home/alexgre/projects/transformer_pretrained_models/345m_uf_syn_pubmed_mimic_wiki_fullcased50k_megatronv22_release \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 256 \
		--cache_data \
		--do_train \
		--do_eval \
		--do_lower_case \
		--train_batch_size 4 \
		--eval_batch_size 4 \
		--learning_rate 1e-5 \
		--num_train_epochs 20 \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file $log \
