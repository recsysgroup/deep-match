#!/usr/bin/env bash

export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export CHECKPOINT_PATH=oss://jiusheng-tmp/onion_reduce_v1_20190310
#export SAVEDMODEL_PATH=oss://jiusheng-tmp/fm_model_exp_v16/savedmodel
export SAVEDMODEL_PATH=hdfs://na61storage/pora/na61hunbu/pai_model/fm_model_exp_v16
export DS=20190306

tar -czf /tmp/deep-match.tar.gz .

odpscmd -e 'use etao_backend;
pai -name tensorflow140
-Dscript="file:///tmp/deep-match.tar.gz"
-DentryFile="main.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-DuserDefinedParameters="--task_type=user_save_model --ds='${DS}' --config=scripts/onion_bpr_reduce_v1_config.json --input_path='${CHECKPOINT_PATH}' --output_path='${SAVEDMODEL_PATH}' --model_dir=oss://jiusheng-tmp/fm_model"
'
