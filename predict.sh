#!/usr/bin/env bash

export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export DS=20190326
export INPUT_TABLE=odps://etao_backend/tables/unirec_match_onion_v2_neg_test_point/ds=${DS}
export OUTPUT_TABLE=odps://etao_backend/tables/unirec_match_onion_v2_test_score/ds=${DS}
export MODEL_DIR=oss://jiusheng-tmp/onion_reduce_v2_20190325

tar -czf /tmp/deep-match.tar.gz .

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="main.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-Dtables="'${INPUT_TABLE}'"
-Doutputs="'${OUTPUT_TABLE}'"
-Dcluster="{\"worker\":{\"count\":32}}"
-DuserDefinedParameters="--task_type=predict --config=scripts/onion_reduce_v2_config.json --input_table='${INPUT_TABLE}' --output_table='${OUTPUT_TABLE}' --model_dir='${MODEL_DIR}' --tmp_dir=oss://jiusheng-tmp/fm_tmp --predict_batch_size=128"
'