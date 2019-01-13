import tensorflow as tf
import argparse
import sys

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


def input_fn_builder(table, config):
    def _decode_record(*line):
        ret_dict = {}

        for fea in config.get('user') + config.get('content'):
            ret_dict[fea.get('name')] = line[fea.get('col_index')]

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


def model_fn_builder(config):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        labels = features['label']

        embedding_dim = 64

        hash_dict = {}
        embedding_dict = {}
        for fea in config.get('user') + config.get('content'):
            fea_name = fea.get('name')
            fea_size = fea.get('size')

            hash_dict[fea.get('name')] = tf.string_to_hash_bucket_fast(
                features.get(fea_name),
                fea_size)

            embedding_dict[fea.get('name')] = tf.get_variable(
                fea_name + '_embedding',
                [fea_size, embedding_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.01))

        def _emb_sum(name):
            emb_sum = None
            for fea in config.get(name):
                fea_name = fea.get('name')
                emb = tf.nn.embedding_lookup(embedding_dict.get(fea_name), hash_dict.get(fea_name))
                if emb_sum is None:
                    emb_sum = emb
                else:
                    emb_sum = emb_sum + emb

            return emb_sum

        user_emb = _emb_sum('user')
        content_emb = _emb_sum('content')

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
    FLAGS.worker_count -= 1
    if FLAGS.task_index > 0:
        FLAGS.task_index -= 1

    config = {
        'user': [
            # {
            #     'name': 'user_id',
            #     'size': 1000000,
            #     'col_index': 0,
            # },
            {
                'name': 'user_age',
                'size': 100,
                'col_index': 6,
            },
            {
                'name': 'user_gender',
                'size': 100,
                'col_index': 7,
            },
            {
                'name': 'user_purchase',
                'size': 100,
                'col_index': 8,
            }
        ],
        'content': [
            {
                'name': 'content_id',
                'size': 1000000,
                'col_index': 1,
            },
            {
                'name': 'cate_id',
                'size': 100000,
                'col_index': 5,
            }
        ],
    }

    train_input_fn = input_fn_builder(FLAGS.train_table, config)
    eval_input_fn = input_fn_builder(FLAGS.eval_table, config)
    # predict_input_fn = input_fn_builder()

    model_fn = model_fn_builder(config)

    config = tf.ConfigProto(allow_soft_placement=True)
    distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)
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
