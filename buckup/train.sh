#!/usr/bin/env bash

# https://pre-logview.alibaba-inc.com
export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export TRAIN_TABLE=odps://etao_backend/tables/dataset_movie_lens_1m_train
export EVAL_TABLE=odps://etao_backend/tables/dataset_movie_lens_1m_eval
export TRAIN_POS_TABLE=odps://etao_backend/tables/dataset_movie_lens_1m_pos_train
export EVAL_POS_TABLE=odps://etao_backend/tables/dataset_movie_lens_1m_pos_eval
export TRAIN_NEG_TABLE=odps://etao_backend/tables/dataset_movie_lens_1m_train_neg
export NEG_TABLE=odps://etao_backend/tables/dataset_movie_lens_1m_neg
export MODEL_DIR=oss://jiusheng-tmp/fm_model_movie_len_1m_as_v6

tar -czf /tmp/deep-match.tar.gz .

echo 'use etao_backend;pai -name tensorboard -DsummaryDir="'${MODEL_DIR}'/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"'

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="main.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-Dtables="'${TRAIN_TABLE}','${EVAL_TABLE}','${TRAIN_POS_TABLE}','${EVAL_POS_TABLE}','${TRAIN_NEG_TABLE}','${NEG_TABLE}'"
-Dcluster="{\"ps\":{\"count\":1},\"worker\":{\"count\":16}}"
-DuserDefinedParameters="--task_type=train --config=scripts/movie_len_1m_attention_sum_config.json --train_max_step=1000000 --learning_rate=3e-4 --train_table='${TRAIN_TABLE}' --train_pos_table='${TRAIN_POS_TABLE}' --train_neg_table='${TRAIN_NEG_TABLE}' --eval_table='${EVAL_TABLE}' --eval_pos_table='${EVAL_POS_TABLE}' --eval_neg_table='${NEG_TABLE}' --model_dir='${MODEL_DIR}' --tmp_dir=oss://jiusheng-tmp/fm_tmp --train_batch_size=128"
'