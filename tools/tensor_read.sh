#!/usr/bin/env bash


export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
tar -czf /tmp/deep-match-tensor_read.tar.gz .

odpscmd -e 'use etao_backend;
pai -name tensorflow140_lite -project algo_public_dev
-Dscript="file:///tmp/deep-match-tensor_read.tar.gz"
-DentryFile="tensor_read.py"
-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
-DuserDefinedParameters="--checkpoint=oss://jiusheng-tmp/fm_model_exp_10_day/model.ckpt-100224 --name=user_gender_emb/user_gender_emb"
'
#oss://jiusheng-tmp/fm_model_exp_10_day/model.ckpt-500026