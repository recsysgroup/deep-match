#!/usr/bin/env bash

export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export INPUT_TABLE=odps://etao_backend/tables/buyershow_item_for_embedding
export OUTPUT_TABLE=odps://etao_backend/tables/buyershow_item_embedding

tar -czf /tmp/deep-match.tar.gz ../deep-match

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="deep-match/main.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-Dtables="'${INPUT_TABLE}'"
-Doutputs="'${OUTPUT_TABLE}'"
-Dcluster="{\"ps\":{\"count\":1},\"worker\":{\"count\":4}}"
-DuserDefinedParameters="--task_type=item_embedding --input_table='${INPUT_TABLE}' --output_table='${OUTPUT_TABLE}' --model_dir=oss://jiusheng-tmp/fm_model --tmp_dir=oss://jiusheng-tmp/fm_tmp --predict_batch_size=128"
'