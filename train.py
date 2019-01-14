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


def input_fn_builder(table, config):
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

        return d

    return input_fn


class MatchNet(object):
    def __init__(self, config):
        self.config = config

        embedding_dim = 64

        embedding_dict = {}
        for fea in config.get('user') + config.get('item'):
            fea_name = fea.get('name')
            fea_size = fea.get('size')
            fea_type = fea.get('type')
            if fea_type in ('one_hot', 'seq'):
                embedding_dict[fea.get('name')] = tf.get_variable(
                    fea_name + '_embedding',
                    [fea_size, embedding_dim],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))

        self.embedding_dict = embedding_dict

    def _emb_sum(self, name, features):
        emb_sum = None
        for fea in self.config.get(name):
            fea_name = fea.get('name')
            fea_size = fea.get('size')
            fea_type = fea.get('type')

            emb = None
            if fea_type == 'one_hot':
                hash = tf.string_to_hash_bucket_fast(features.get(fea_name), fea_size, name=fea_name + '_hash')
                emb = tf.nn.embedding_lookup(self.embedding_dict.get(fea_name), hash)

            if fea_type == 'seq':
                hash = tf.string_to_hash_bucket_fast(features.get(fea_name), fea_size, name=fea_name + '_hash')
                mask = tf.expand_dims(tf.cast(features.get(fea_name + '_mask'), dtype=tf.float32), axis=-1)
                emb = tf.nn.embedding_lookup(self.embedding_dict.get(fea_name), hash) * mask
                emb = tf.reduce_sum(emb, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-6)

            if emb_sum is None:
                emb_sum = emb
            else:
                emb_sum = emb_sum + emb

        return emb_sum

    def user_embedding(self, features):
        return self._emb_sum('user', features)

    def item_embedding(self, features):
        return self._emb_sum('item', features)


def model_fn_builder(config):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        labels = features['label']

        matchNet = MatchNet(config)

        user_emb = matchNet.user_embedding(features)
        content_emb = matchNet.item_embedding(features)

        predictions = tf.reduce_sum(user_emb * content_emb, axis=1)

        loss = tf.losses.sigmoid_cross_entropy(labels, predictions)
        # loss += 0.1 * tf.reduce_mean(tf.reduce_sum(tf.square(user_emb), axis=1))
        # loss += 0.1 * tf.reduce_mean(tf.reduce_sum(tf.square(content_emb), axis=1))

        if mode == tf.estimator.ModeKeys.TRAIN:

            global_step = tf.train.get_or_create_global_step()

            learning_rate = tf.constant(value=params["learning_rate"], shape=[], dtype=tf.float32)

            # Implements linear decay of the learning rate.
            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                FLAGS.train_max_step,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False)

            opt = tf.train.AdamOptimizer(learning_rate)
            train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops={
                    "eval_loss": tf.metrics.mean(loss),
                    "auc": tf.metrics.auc(labels, tf.sigmoid(predictions))
                }
            )

        else:
            predictions = None
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def test_input_fn(_input_fn):
    iterator = _input_fn().make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))


def main(_):
    print(FLAGS.task_index, FLAGS.job_name, FLAGS.worker_count)
    if FLAGS.worker_count > 1:
        FLAGS.worker_count -= 1
    if FLAGS.task_index > 0:
        FLAGS.task_index -= 1

    config = {
        'embedding_size': 64,  # user and item final dim size
        'user': [
            # {
            #     'name': 'user_id',
            #     'size': 1000000,
            #     'value_index': 0,
            # },
            # {
            #     'name': 'user_age',
            #     'size': 100,
            #     'value_index': 6,
            #     'need_hash': True,
            #     'type': 'one_hot'
            # },
            # {
            #     'name': 'user_gender',
            #     'size': 100,
            #     'value_index': 7,
            #     'need_hash': True,
            #     'type': 'one_hot'
            # },
            # {
            #     'name': 'user_purchase',
            #     'size': 100,
            #     'value_index': 8,
            #     'need_hash': True,
            #     'type': 'one_hot'  # one_hot,dense,
            # },
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
            }
        ],
    }

    train_input_fn = input_fn_builder(FLAGS.train_table, config)
    eval_input_fn = input_fn_builder(FLAGS.eval_table, config)
    # predict_input_fn = input_fn_builder()

    model_fn = model_fn_builder(config)

    config = tf.ConfigProto(allow_soft_placement=True)
    distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)
    # distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        session_config=config,
        distribute=distribution,
        save_checkpoints_steps=50000,
    )

    params = {
        "learning_rate": FLAGS.learning_rate,
        "batch_size": FLAGS.train_batch_size
    }

    estimator = tf.estimator.Estimator(
        model_dir=FLAGS.model_dir,
        model_fn=model_fn,
        params=params,
        config=run_config
    )

    if FLAGS.do_train:
        print('do_train')
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_max_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=300)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
