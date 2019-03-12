import tensorflow as tf
import argparse
import sys
import numpy as np
import json
import constant as C
from modeling import MatchNet
from helper import LogviewMetricHook, LogviewMetricWriter, LogviewTrainHook, get_assignment_map_from_checkpoint
from losses import build_loss_fn
from evals import build_eval_fn

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
flags.DEFINE_string("init_checkpoint", None, "")
flags.DEFINE_string("model_dir", None, "")
flags.DEFINE_integer("train_batch_size", 128, "")
flags.DEFINE_integer("eval_batch_size", 128, "")
flags.DEFINE_integer("predict_batch_size", 1000, "")
flags.DEFINE_integer("train_max_step", 1000, "")
flags.DEFINE_boolean("do_train", True, "")
flags.DEFINE_boolean("do_predict", False, "")
flags.DEFINE_float("learning_rate", 1e-4, "")
flags.DEFINE_string("config", None, "")

# task
flags.DEFINE_string("train_pos_table", None, "")
flags.DEFINE_string("eval_pos_table", None, "")
flags.DEFINE_string("train_neg_table", None, "")
flags.DEFINE_string("eval_neg_table", None, "")
flags.DEFINE_string("task_type", None, "train,user_embedding,item_embedding,predict")
flags.DEFINE_boolean("drop_out", True, "")
flags.DEFINE_string("input_path", None, "")
flags.DEFINE_string("output_path", None, "")
flags.DEFINE_string("ds", None, "")

logviewMetricWriter = LogviewMetricWriter(FLAGS.model_dir)


def parse_seq_str_op(text, max_seq_len):
    def user_func(text):
        vals = []
        mask = []
        parts = text.split(',')
        for i in range(min(len(parts), max_seq_len)):
            vals.append(parts[i])
            mask.append(1.0)

        vals = vals + ['0'] * (max_seq_len - len(vals))
        mask = mask + [0.0] * (max_seq_len - len(mask))

        return vals, mask

    y = tf.py_func(user_func, [text], [tf.string, tf.double])
    y[0].set_shape((max_seq_len))
    y[1].set_shape((max_seq_len))
    return y


def parse_dense_str_op(text, max_len):
    def user_func(text):
        vals = []
        if text != '':
            parts = text.split(',')
            for i in range(min(len(parts), max_len)):
                vals.append(float(parts[i]))

        vals = vals + [0.0] * (max_len - len(vals))

        return np.array(vals, dtype=np.float32)

    y = tf.py_func(user_func, [text], tf.float32)
    y.set_shape((max_len))
    return y


def table_dataset_and_decode_builder(table, config, extra_fields=[],
                                     fea_sides=['user', 'item'],
                                     slice_id=FLAGS.task_index,
                                     slice_count=FLAGS.worker_count):
    columns = []
    for fea_side in fea_sides:
        columns.append(('__' + fea_side + '__', 'single', {}))

    for column in config.get(C.CONFIG_COLUMNS):
        col_side, col_name = column.get(C.CONFIG_COLUMNS_NAME).split(':')
        col_type = column.get(C.CONFIG_COLUMNS_TYPE)
        if col_side in fea_sides:
            columns.append((col_name, col_type, column))

    record_defaults = []
    selected_cols = []
    colname_2_index = {}

    for column in columns:
        record_defaults.append('')
        selected_cols.append(column[0])
        colname_2_index[column[0]] = len(colname_2_index)

    for field in extra_fields:
        selected_cols.append(field[0])
        record_defaults.append(field[1])
        colname_2_index[field[0]] = len(colname_2_index)

    def _decode_record(*line):
        ret_dict = {}

        for name, type, col in columns:
            if type == C.CONFIG_COLUMNS_TYPE_SINGLE:
                ret_dict[name] = line[colname_2_index.get(name)]
            elif type == C.CONFIG_COLUMNS_TYPE_SEQ:
                _value, _mask = parse_seq_str_op(line[colname_2_index.get(name)], col.get('seq_len'))
                ret_dict[name] = _value
                ret_dict[name + '_mask'] = _mask
            elif type == C.CONFIG_COLUMNS_TYPE_DENSE:
                _value = parse_dense_str_op(line[colname_2_index.get(name)], col.get(C.CONFIG_COLUMNS_TYPE_DENSE_SIZE))
                ret_dict[name] = _value

        for field in extra_fields:
            ret_dict[field[0]] = line[colname_2_index.get(field[0])]

        return ret_dict

    tf.logging.info("read table [{0}], record_defaults is {1}, selected_cols is {2}".format(table, str(record_defaults),
                                                                                            str(selected_cols)))
    d = tf.data.TableRecordDataset([table],
                                   record_defaults=record_defaults,
                                   slice_id=slice_id,
                                   slice_count=slice_count,
                                   selected_cols=','.join(selected_cols),
                                   )

    return d, _decode_record


def predict_input_fn_builder(table, config):
    def input_fn():
        if FLAGS.task_type == C.TASK_TYPE_ITEM_EMBEDDING:
            dataset, decode = table_dataset_and_decode_builder(table, config,
                                                               fea_sides=['item'])
        elif FLAGS.task_type == C.TASK_TYPE_USER_EMBEDDING:
            dataset, decode = table_dataset_and_decode_builder(table, config,
                                                               fea_sides=['user'])

        dataset = dataset.repeat(1)
        dataset = dataset.map(decode, num_parallel_calls=C.NUM_PARALLEL_CALLS)
        dataset = dataset.batch(batch_size=FLAGS.predict_batch_size)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element

    return input_fn


def eval_input_fn_builder(config):
    def input_fn():
        _zip = {}
        if FLAGS.eval_table:
            rank_dataset, rank_decode = table_dataset_and_decode_builder(FLAGS.eval_table, config,
                                                                         extra_fields=[('label', 0)], slice_id=0,
                                                                         slice_count=1)
            rank_dataset = rank_dataset.repeat(1)
            rank_dataset = rank_dataset.map(rank_decode, num_parallel_calls=C.NUM_PARALLEL_CALLS)
            rank_dataset = rank_dataset.batch(batch_size=config.get(C.CONFIG_EVAL_POINT_SIZE))
            _zip[C.CONFIG_INPUT_POINT_FEATURES] = rank_dataset

        if FLAGS.eval_pos_table:
            match_dataset, match_decode = table_dataset_and_decode_builder(FLAGS.eval_pos_table, config,
                                                                           slice_id=0,
                                                                           slice_count=1)
            match_dataset = match_dataset.repeat(1)
            match_dataset = match_dataset.map(match_decode, num_parallel_calls=C.NUM_PARALLEL_CALLS)
            match_dataset = match_dataset.batch(batch_size=config.get(C.CONFIG_EVAL_POS_SIZE))
            _zip[C.CONFIG_INPUT_POS_FEATURES] = match_dataset

        if FLAGS.eval_neg_table:
            neg_dataset, neg_decode = table_dataset_and_decode_builder(FLAGS.eval_neg_table, config, fea_sides=['item'],
                                                                       slice_id=0, slice_count=1, )
            neg_dataset = neg_dataset.repeat(1)
            neg_dataset = neg_dataset.shuffle(buffer_size=100000)
            neg_dataset = neg_dataset.map(neg_decode, num_parallel_calls=C.NUM_PARALLEL_CALLS)
            neg_dataset = neg_dataset.batch(batch_size=config.get(C.CONFIG_EVAL_NEG_SIZE))
            _zip[C.CONFIG_INPUT_NEG_FEATURES] = neg_dataset

        return tf.data.Dataset.zip(_zip)

    return input_fn


def train_input_fn_builder(config):
    def input_fn():
        _zip = {}

        if FLAGS.train_table:
            rank_dataset, rank_decode = table_dataset_and_decode_builder(FLAGS.train_table, config, [('label', 0)])
            rank_dataset = rank_dataset.repeat()
            rank_dataset = rank_dataset.shuffle(buffer_size=10000)
            rank_dataset = rank_dataset.map(rank_decode, num_parallel_calls=C.NUM_PARALLEL_CALLS)
            rank_dataset = rank_dataset.batch(batch_size=config.get(C.CONFIG_TRAIN_BATCH_SIZE))
            rank_dataset = rank_dataset.prefetch(buffer_size=config.get(C.CONFIG_TRAIN_BATCH_SIZE))
            _zip[C.CONFIG_INPUT_POINT_FEATURES] = rank_dataset

        if FLAGS.train_pos_table:
            match_dataset, match_decode = table_dataset_and_decode_builder(FLAGS.train_pos_table, config)
            match_dataset = match_dataset.repeat()
            match_dataset = match_dataset.shuffle(buffer_size=10000)
            match_dataset = match_dataset.map(match_decode, num_parallel_calls=C.NUM_PARALLEL_CALLS)
            match_dataset = match_dataset.batch(batch_size=config.get(C.CONFIG_TRAIN_BATCH_SIZE))
            match_dataset = match_dataset.prefetch(buffer_size=config.get(C.CONFIG_TRAIN_BATCH_SIZE))
            _zip[C.CONFIG_INPUT_POS_FEATURES] = match_dataset

        if FLAGS.train_neg_table:
            neg_dataset, neg_decode = table_dataset_and_decode_builder(FLAGS.train_neg_table, config,
                                                                       fea_sides=['item'],
                                                                       slice_id=0, slice_count=1, )
            neg_dataset = neg_dataset.repeat()
            neg_dataset = neg_dataset.shuffle(buffer_size=10000)
            neg_dataset = neg_dataset.map(neg_decode, num_parallel_calls=C.NUM_PARALLEL_CALLS)
            neg_dataset = neg_dataset.batch(batch_size=config.get(C.CONFIG_TRAIN_NEG_SIZE))
            neg_dataset = neg_dataset.prefetch(buffer_size=config.get(C.CONFIG_TRAIN_NEG_SIZE))
            _zip[C.CONFIG_INPUT_NEG_FEATURES] = neg_dataset

        return tf.data.Dataset.zip(_zip)

    return input_fn


def model_fn_builder(config):
    def _match_loss_and_metric(matchNet, config, features):
        match_features = features.get('match_features')

        user_emb_for_match = matchNet.user_embedding(match_features)
        content_emb_for_match = matchNet.item_embedding(match_features)

        neg_size = config.get('neg_sample_size')

        tmp_emb = tf.tile(content_emb_for_match, [1, 1])
        for i in range(neg_size):
            rand = int((1 + i) * FLAGS.train_batch_size / (neg_size + 1))
            content_emb_for_match = tf.concat([content_emb_for_match,
                                               tf.slice(tmp_emb, [rand, 0], [FLAGS.train_batch_size - rand, -1]),
                                               tf.slice(tmp_emb, [0, 0], [rand, -1])], 0)

        point_multi = tf.reduce_sum(tf.tile(user_emb_for_match, [neg_size + 1, 1]) * content_emb_for_match, 1,
                                    True)
        dis = tf.transpose(tf.reshape(tf.transpose(point_multi), [neg_size + 1, FLAGS.train_batch_size]))

        prob = tf.nn.softmax(dis)
        match_loss = -tf.reduce_sum(tf.log(tf.slice(prob, [0, 0], [-1, 1]))) / FLAGS.train_batch_size

        return match_loss, tf.metrics.mean(metric_mrr(prob, neg_size))

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        if mode == tf.estimator.ModeKeys.TRAIN:
            matchNet = MatchNet(config, True)
            _loss_params = config.get(C.CONFIG_LOSS)
            _loss_name = _loss_params.get(C.CONFIG_LOSS_NAME)
            loss_fn = build_loss_fn(_loss_name, _loss_params)
            _loss = loss_fn(matchNet, features)

            if FLAGS.init_checkpoint:
                tvars = tf.trainable_variables()
                assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars,
                                                                                                FLAGS.init_checkpoint)
                tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
                tf.logging.info("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)

            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.constant(value=params["learning_rate"], shape=[], dtype=tf.float32)

            # # Implements linear decay of the learning rate.
            # learning_rate = tf.train.polynomial_decay(
            #     learning_rate,
            #     global_step,
            #     FLAGS.train_max_step,
            #     end_learning_rate=0.0,
            #     power=1.0,
            #     cycle=False)

            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                1000,
                end_learning_rate=1e-4,
                power=1.0,
                cycle=True)

            # learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.99)

            opt = tf.train.AdamOptimizer(learning_rate)
            train_op = opt.minimize(_loss, global_step=tf.train.get_global_step())
            # hook = tf.train.ProfilerHook(save_steps=10000, output_dir=FLAGS.tmp_dir)
            tf.summary.scalar('learning_rate', learning_rate)
            train_hook = LogviewTrainHook(_loss, learning_rate, global_step, logviewMetricWriter)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=_loss,
                train_op=train_op,
                training_chief_hooks=[train_hook]
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            matchNet = MatchNet(config, False)
            # _loss_params = config.get(C.CONFIG_LOSS)
            # _loss_name = _loss_params.get(C.CONFIG_LOSS_NAME)
            # loss_fn = build_loss_fn(_loss_name, _loss_params)
            # _loss = loss_fn(matchNet, features)
            eval_metric_ops = {}  # {'eval_loss': _loss}

            for _eval in config.get(C.CONFIG_EVALS):
                _eval_name = _eval.get(C.CONFIG_EVALS_NAME)
                eval_fn = build_eval_fn(_eval_name, _eval)
                _metric = eval_fn(matchNet, features)
                eval_metric_ops[_eval_name] = _metric
                if type(_metric) is tuple:
                    tf.summary.scalar(_eval_name, _metric[1])
                elif type(_metric) is dict:
                    for k, v in _metric.items():
                        tf.summary.scalar(k, v)
                else:
                    tf.summary.scalar(_eval_name, _metric)

            global_step = tf.train.get_or_create_global_step()

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.constant(0, dtype=tf.float32),
                eval_metric_ops={},
                evaluation_hooks=[LogviewMetricHook(eval_metric_ops, global_step, logviewMetricWriter)]
            )

        else:
            matchNet = MatchNet(config, False)
            predictions = {}
            predict_type = FLAGS.task_type
            if predict_type == C.TASK_TYPE_USER_EMBEDDING:
                predictions['__user__'] = features.get('__user__')
                user_embedding = matchNet.user_embedding(features)
                user_embedding = tf.reduce_join(tf.as_string(user_embedding, precision=5), 1, separator=',')
                predictions['user_embedding'] = user_embedding
            elif predict_type == C.TASK_TYPE_ITEM_EMBEDDING:
                predictions['__item__'] = features.get('__item__')
                item_embedding = matchNet.item_embedding(features)
                item_embedding = tf.reduce_join(tf.as_string(item_embedding, precision=5), 1, separator=',')
                predictions['item_embedding'] = item_embedding
            elif predict_type == C.TASK_TYPE_PREDICT:
                pass

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def export_saved_model(config):
    matchNet = MatchNet(config, False)

    features = {}
    input_infos = {}
    features['__user__'] = tf.placeholder(tf.string, shape=[None], name='user_bias')
    input_infos['__user__'] = tf.saved_model.utils.build_tensor_info(features.get('__user__'))

    for column in config.get(C.CONFIG_COLUMNS):
        fea_side, fea_name = column.get(C.CONFIG_COLUMNS_NAME).split(':')
        fea_type = column.get(C.CONFIG_COLUMNS_TYPE)

        if fea_side == 'user':
            if fea_type == C.CONFIG_COLUMNS_TYPE_SINGLE:
                features[fea_name] = tf.placeholder(tf.string, shape=[None], name=fea_name)
            elif fea_type == C.CONFIG_COLUMNS_TYPE_SEQ:
                seq_len = column.get('seq_len')
                features[fea_name] = tf.placeholder(tf.string, shape=[None, seq_len], name=fea_name)
                features[fea_name + "_mask"] = tf.placeholder(tf.float32, shape=[None, seq_len],
                                                              name=fea_name + '_mask')

            input_infos[fea_name] = tf.saved_model.utils.build_tensor_info(features.get(fea_name))

    user_emb = matchNet.user_embedding(features)

    user_emb_output = tf.identity(user_emb, name="user_embedding")

    signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=input_infos,
            outputs={
                "user_embedding":
                    tf.saved_model.utils.build_tensor_info(
                        user_emb_output)
            },
            method_name="embedding")
    )

    saver = tf.train.Saver()

    version = 0
    path = '{0}/{1}/data'.format(FLAGS.output_path, FLAGS.ds)
    while tf.gfile.Exists(path):
        version += 1
        path = '{0}/{1}{2}/data'.format(FLAGS.output_path, FLAGS.ds, str(version))

    print('export path : {0}'.format(path))

    builder = tf.saved_model.builder.SavedModelBuilder(path)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        print tf.train.latest_checkpoint(FLAGS.input_path)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.input_path))
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"signature": signature})
    builder.save()


def main(_):
    with open(FLAGS.config, 'r') as f:
        config = json.load(f)

    model_fn = model_fn_builder(config)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)
    # distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        session_config=session_config,
        distribute=distribution,
        save_checkpoints_steps=50000,
        keep_checkpoint_max=10,
    )

    params = {
        "learning_rate": FLAGS.learning_rate,
        "batch_size": FLAGS.train_batch_size,
    }

    estimator = tf.estimator.Estimator(
        model_dir=FLAGS.model_dir,
        model_fn=model_fn,
        params=params,
        config=run_config,
    )

    print(json.dumps(config))
    from multiprocessing import cpu_count
    print("cpu count is {0}".format(cpu_count()))

    if FLAGS.task_type == C.TASK_TYPE_TRAIN:
        print(FLAGS.task_index, FLAGS.job_name, FLAGS.worker_count)
        if FLAGS.task_index == 1:
            FLAGS.drop_out = False
        if FLAGS.worker_count > 1:
            FLAGS.worker_count -= 1
        if FLAGS.task_index > 0:
            FLAGS.task_index -= 1
        train_input_fn = train_input_fn_builder(config)
        eval_input_fn = eval_input_fn_builder(config)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_max_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=120, throttle_secs=60)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.task_type in (C.TASK_TYPE_USER_EMBEDDING, C.TASK_TYPE_ITEM_EMBEDDING):
        predict_input_fn = predict_input_fn_builder(FLAGS.input_table, config)

        writer = tf.python_io.TableWriter(FLAGS.output_table, slice_id=FLAGS.task_index)

        _writen_num = 0
        for result in estimator.predict(input_fn=predict_input_fn):
            _writen_num += 1
            if _writen_num % 10000 == 0:
                print('{0} records have been writen'.format(_writen_num))
            if FLAGS.task_type == C.TASK_TYPE_USER_EMBEDDING:
                writer.write([result.get('__user__'), result.get('user_embedding')], [0, 1])
            elif FLAGS.task_type == C.TASK_TYPE_ITEM_EMBEDDING:
                writer.write([result.get('__item__'), result.get('item_embedding')], [0, 1])

        writer.close()

    if FLAGS.task_type in [C.TASK_TYPE_USER_SAVE_MODEL]:
        export_saved_model(config)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
