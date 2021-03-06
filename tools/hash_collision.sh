#!/usr/bin/env bash

export TABLE=odps://etao_backend/tables/unirec_match_onion_v1_train_point/ds=20190222

tar -czf /tmp/deep-match-hash_collision.tar.gz .

odpscmd -e 'use etao_backend;
pai -name tensorflow140_lite -project algo_public_dev
-Dscript="file:///tmp/deep-match-hash_collision.tar.gz"
-DentryFile="hash_collision.py"
-Dtables="'${TABLE}'"
-DuserDefinedParameters="--table='${TABLE}' --field=cate_id --size=200000"
'
