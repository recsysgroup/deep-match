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

        for fea in config.get('user') + config.get('item'):
            fea_name = fea.get('name')
            fea_type = fea.get('type')
            value_index = fea.get('value_index')

            if fea_type == 'one_hot':
                ret_dict[fea_name] = line[value_index]

            if fea_type == 'seq':
                value_and_mask = parse_seq_str_op(line[value_index], fea.get('seq_len'))
                ret_dict[fea_name] = value_and_mask[0]
                ret_dict[fea_name + '_mask'] = value_and_mask[1]

        ret_dict['label'] = line[2]

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
                                       record_defaults=_recode_defaults,
                                       slice_id=FLAGS.task_index,
                                       slice_count=FLAGS.worker_count)

        d = d.repeat()
        d = d.shuffle(buffer_size=1000)
        d = d.map(_decode_record)
        d = d.batch(batch_size=FLAGS.train_batch_size)

        match_dataset = tf.data.TableRecordDataset([match_table],
                                                   record_defaults=_recode_defaults,
                                                   slice_id=FLAGS.task_index,
                                                   slice_count=FLAGS.worker_count)
        match_dataset = match_dataset.repeat()
        match_dataset = match_dataset.shuffle(buffer_size=10000)
        match_dataset = match_dataset.map(_decode_record)
        match_dataset = match_dataset.batch(batch_size=FLAGS.train_batch_size)

        return tf.data.Dataset.zip({'match_features': match_dataset, 'rank_features': d})

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


    match_features = features.get('match_features')
    print match_features
    user_id = match_features.get('user_id')
    content_id = match_features.get('content_id')
    print content_id.get_shape()
    print user_id.get_shape()

    neg_size = config.get('neg_sample_size')

    tmp_emb = tf.tile(content_id, [1])
    for i in range(neg_size):
        rand = int((1 + i) * FLAGS.train_batch_size / (neg_size + 1))
        content_id = tf.concat([content_id,
                                tf.slice(tmp_emb, [rand], [FLAGS.train_batch_size - rand]),
                                tf.slice(tmp_emb, [0], [rand])], 0)

    user_id = tf.tile(user_id, [neg_size])
    content_id = tf.slice(content_id, [FLAGS.train_batch_size], [-1])

    writer = tf.python_io.TableWriter(FLAGS.output_table, slice_id=FLAGS.task_index)

    with tf.Session() as sess:
        for i in range(1000):
            _user_id, _content_id = sess.run([user_id, content_id])
            print _user_id, _content_id
            for j in range(len(_user_id)):
                writer.write([_user_id[j], _content_id[j]], [0, 1])

    writer.close()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
