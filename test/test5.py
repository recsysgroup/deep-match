import tensorflow as tf
import argparse
import sys
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

# pai
flags.DEFINE_integer("task_index", 0, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("worker_count", 1, "")

# common
flags.DEFINE_string("train_table", None, "")
flags.DEFINE_string("eval_table", None, "")
flags.DEFINE_string("input_table", None, "")
flags.DEFINE_string("output_table", None, "")
flags.DEFINE_string("tmp_dir", None, "")
flags.DEFINE_string("model_dir", None, "")
flags.DEFINE_integer("train_batch_size", 128, "")
flags.DEFINE_integer("train_max_step", 1000, "")
flags.DEFINE_boolean("do_train", True, "")
flags.DEFINE_boolean("do_predict", False, "")
flags.DEFINE_float("learning_rate", 1e-4, "")

# task
flags.DEFINE_string("input_emb_len", None, "")
flags.DEFINE_string("output_emb_len", None, "")
flags.DEFINE_string("train_match_table", None, "")
flags.DEFINE_string("eval_match_table", None, "")


def parse_seq_str_op(text, max_seq_len):
    def user_func(text):
        ids = []
        mask = []
        parts = text.split(',')
        for i in range(min(len(parts), max_seq_len)):
            ids.append(parts[i])
            mask.append(1.0)

        ids = ids + [''] * (max_seq_len - len(ids))
        mask = mask + [0.0] * (max_seq_len - len(mask))

        return ids, mask

    y = tf.py_func(user_func, [text], [tf.string, tf.double])
    y[0].set_shape((max_seq_len))
    y[1].set_shape((max_seq_len))
    return y


def input_fn_builder(match_table, table, config):
    def _decode_record(*line):
        ret_dict = {}

        ret_dict['label'] = line[0]
        ret_dict['content_id'] = line[1]

        return ret_dict

    _recode_defaults = (''
                        , ''
                        , 0
                        , 0
                        , ''
                        , ''
                        , ''
                        , ''
                        , ''
                        , ''
                        )

    def input_fn():
        d = tf.data.TableRecordDataset([table],
                                       record_defaults=(0, ''),
                                       slice_id=FLAGS.task_index,
                                       slice_count=FLAGS.worker_count,
                                       selected_cols='label,content_id')

        d = d.repeat()
        d = d.shuffle(buffer_size=1000)
        d = d.map(_decode_record)
        d = d.batch(batch_size=FLAGS.train_batch_size)

        return d

    return input_fn


def test_input_fn(_input_fn):
    iterator = _input_fn().make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))


def main(_):
    print(FLAGS.task_index, FLAGS.job_name, FLAGS.worker_count)

    config = {
        'embedding_size': 64,  # user and item final dim size
        'neg_sample_size': 5,
        'user': [
            {
                'name': 'user_id',
                'size': 1000000,
                'value_index': 0,
                'need_hash': True,
                'type': 'one_hot'
            },
            {
                'name': 'user_age',
                'size': 100,
                'value_index': 6,
                'need_hash': True,
                'type': 'one_hot'
            },
            {
                'name': 'user_gender',
                'size': 100,
                'value_index': 7,
                'need_hash': True,
                'type': 'one_hot'
            },
            {
                'name': 'user_purchase',
                'size': 100,
                'value_index': 8,
                'need_hash': True,
                'type': 'one_hot'  # one_hot,dense,
            },
            {
                'name': 'user_rootcates',
                'size': 10000,
                'need_hash': True,
                'value_index': 9,
                'type': 'seq',
                'seq_len': 5,  # if type=seq, need this param
                # 'weight_index': 444,  # if equal weights, this param could be None
                # 'time_index': 555,  # if it has time embedding,the value should be int type
                'reduce_method': 'mean',  # mean,sum,mlp,attention,rnn,cnn
                # 'layer_size': 2 #mlp,attention need this param
            }
        ],
        'item': [
            {
                'name': 'content_id',
                'size': 1000000,
                'value_index': 1,
                'need_hash': True,
                'type': 'one_hot'
            },
            {
                'name': 'cate_id',
                'size': 100000,
                'value_index': 5,
                'need_hash': True,
                'type': 'one_hot'
            },
        ],
    }

    train_input_fn = input_fn_builder(FLAGS.train_match_table, FLAGS.train_table, config)

    iterator = train_input_fn().make_one_shot_iterator()
    features = iterator.get_next()

    with tf.Session() as sess:
        for i in range(10):
            _features = sess.run(features)
            print _features


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
