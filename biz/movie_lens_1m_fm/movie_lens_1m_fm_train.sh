

export TRAIN_TABLE=~/Data/ml-1m/ml_1m_sideinfo_data_train.csv
export TEST_TABLE=~/Data/ml-1m/ml_1m_sideinfo_data_train.csv
export MODEL_DIR=/tmp/movie_lens_1m/v31
export CONFIG=biz/movie_lens_1m_fm/movie_lens_1m_fm_config.json

python2 main.py --task_type=train --train_max_step=1000000 --config=${CONFIG} --train_pos_table=${TRAIN_TABLE} --eval_pos_table=${TEST_TABLE} --model_dir=${MODEL_DIR}
