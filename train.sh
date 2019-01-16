#!/usr/bin/env bash

export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export TRAIN_TABLE=odps://etao_backend/tables/buyershow_dataset_train
export EVAL_TABLE=odps://etao_backend/tables/buyershow_dataset_valid
export TRAIN_MATCH_TABLE=odps://etao_backend/tables/buyershow_dataset_train_match
export EVAL_MATCH_TABLE=odps://etao_backend/tables/buyershow_dataset_valid_match

tar -czf /tmp/deep-match.tar.gz ../deep-match

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="deep-match/train.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-Dtables="'${TRAIN_TABLE}','${EVAL_TABLE}','${TRAIN_MATCH_TABLE}','${EVAL_MATCH_TABLE}'"
-Dcluster="{\"ps\":{\"count\":1},\"worker\":{\"count\":4}}"
-DuserDefinedParameters="--train_max_step=500000 --learning_rate=1e-3 --train_table='${TRAIN_TABLE}' --eval_table='${TRAIN_TABLE}' --train_match_table='${TRAIN_MATCH_TABLE}' --eval_match_table='${TRAIN_MATCH_TABLE}' --model_dir=oss://jiusheng-tmp/fm_model_v2 --tmp_dir=oss://jiusheng-tmp/fm_tmp --train_batch_size=128"
'