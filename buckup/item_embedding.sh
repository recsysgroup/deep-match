#!/usr/bin/env bash

export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export INPUT_TABLE=odps://etao_backend/tables/unirec_match_onion_content_feature/ds=20190214
export OUTPUT_TABLE=odps://etao_backend/tables/buyershow_item_embedding/ds=20190214
export MODEL_DIR=oss://jiusheng-tmp/fm_model_onion_bpr_v2

tar -czf /tmp/deep-match.tar.gz .

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="main.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-Dtables="'${INPUT_TABLE}'"
-Doutputs="'${OUTPUT_TABLE}'"
-Dcluster="{\"ps\":{\"count\":1},\"worker\":{\"count\":8}}"
-DuserDefinedParameters="--task_type=item_embedding --config=scripts/onion_config.json --input_table='${INPUT_TABLE}' --output_table='${OUTPUT_TABLE}' --model_dir='${MODEL_DIR}' --tmp_dir=oss://jiusheng-tmp/fm_tmp --predict_batch_size=128"
'