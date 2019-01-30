#!/usr/bin/env bash

# https://pre-logview.alibaba-inc.com
export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export TRAIN_TABLE=odps://etao_backend/tables/buyershow_dataset_train
export EVAL_TABLE=odps://etao_backend/tables/buyershow_dataset_valid
export TRAIN_MATCH_TABLE=odps://etao_backend/tables/buyershow_dataset_train_match
export EVAL_MATCH_TABLE=odps://etao_backend/tables/buyershow_dataset_valid_match
export TRAIN_NEG_TABLE=odps://etao_backend/tables/buyershow_dataset_neg_sample_valid_v2
export EVAL_NEG_TABLE=odps://etao_backend/tables/buyershow_dataset_neg_sample_valid
export MODEL_DIR=oss://jiusheng-tmp/fm_model_v4

tar -czf /tmp/deep-match.tar.gz ../deep-match

ossutilmac64 -rf $MODEL_DIR

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="deep-match/main.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-Dtables="'${TRAIN_TABLE}','${EVAL_TABLE}','${TRAIN_MATCH_TABLE}','${EVAL_MATCH_TABLE}','${TRAIN_NEG_TABLE}','${EVAL_NEG_TABLE}'"
-Dcluster="{\"ps\":{\"count\":1},\"worker\":{\"count\":4}}"
-DuserDefinedParameters="--task_type=train --train_max_step=500000 --learning_rate=1e-4 --train_table='${TRAIN_TABLE}' --eval_table='${EVAL_TABLE}' --train_match_table='${TRAIN_MATCH_TABLE}' --eval_match_table='${TRAIN_MATCH_TABLE}' --train_neg_table='${EVAL_NEG_TABLE}' --eval_neg_table='${EVAL_NEG_TABLE}' --model_dir='${MODEL_DIR}' --tmp_dir=oss://jiusheng-tmp/fm_tmp --train_batch_size=128"
'