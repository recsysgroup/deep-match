#!/usr/bin/env bash

export EXP_NAME=exp_v1
export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export DS=20190406
export INPUT_TABLE=odps://etao_backend/tables/unirec_onion_item2vec_debug_sample/ds=${DS}
export OUTPUT_TABLE=odps://etao_backend/tables/unirec_onion_item2vec_debug/ds=${DS}
export MODEL_DIR=oss://jiusheng-tmp/fm_model_${EXP_NAME}
export CONFIG=scripts/onion_item2vec_${EXP_NAME}_config.json

tar -czf /tmp/deep-match.tar.gz .

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="main.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-Dtables="'${INPUT_TABLE}'"
-Doutputs="'${OUTPUT_TABLE}'"
-Dcluster="{\"worker\":{\"count\":16}}"
-DuserDefinedParameters="--task_type=debug --config='${CONFIG}' --input_table='${INPUT_TABLE}' --output_table='${OUTPUT_TABLE}' --model_dir='${MODEL_DIR}' --tmp_dir=oss://jiusheng-tmp/fm_tmp --predict_batch_size=128"
'