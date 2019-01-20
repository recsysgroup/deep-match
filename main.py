import tensorflow as tf
import argparse
import sys
import numpy as np
import constant as C
from modeling import MatchNet

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
flags.DEFINE_integer("predict_batch_size", 1000, "")
flags.DEFINE_integer("train_max_step", 1000, "")
flags.DEFINE_boolean("do_train", True, "")
flags.DEFINE_boolean("do_predict", False, "")
flags.DEFINE_float("learning_rate", 1e-4, "")

# task
flags.DEFINE_string("input_emb_len", None, "")
flags.DEFINE_string("output_emb_len", None, "")
flags.DEFINE_string("train_match_table", None, "")
flags.DEFINE_string("eval_match_table", None, "")
flags.DEFINE_string("train_neg_table", None, "")
flags.DEFINE_string("eval_neg_table", None, "")
flags.DEFINE_string("task_type", None, "train,user_embedding,item_embedding,predict")


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


def table_dataset_and_decode_builder(table, config, extra_fields=[],
                                     fea_sides=['user', 'item'],
                                     slice_id=FLAGS.task_index,
                                     slice_count=FLAGS.worker_count):
    feas = []
    for fea_side in fea_sides:
        feas.extend(config.get(fea_side))

    record_defaults = []
    selected_cols = []
    colname_2_index = {}
    for fea in feas:
        record_defaults.append('')
        selected_cols.append(fea.get('name'))
        colname_2_index[fea.get('name')] = len(colname_2_index)

    for field in extra_fields:
        selected_cols.append(field[0])
        record_defaults.append(field[1])
        colname_2_index[field[0]] = len(colname_2_index)

    def _decode_record(*line):
        ret_dict = {}

        for fea in feas:
            fea_name = fea.get('name')
            fea_type = fea.get('type')

            if fea_type == 'one_hot':
                ret_dict[fea_name] = line[colname_2_index.get(fea_name)]

            if fea_type == 'seq':
                value_and_mask = parse_seq_str_op(line[colname_2_index.get(fea_name)], fea.get('seq_len'))
                ret_dict[fea_name] = value_and_mask[0]
                ret_dict[fea_name + '_mask'] = value_and_mask[1]

        for field in extra_fields:
            ret_dict[field[0]] = line[colname_2_index.get(field[0])]

        return ret_dict

    d = tf.data.TableRecordDataset([table],
                                   record_defaults=record_defaults,
                                   slice_id=slice_id,
                                   slice_count=slice_count,
                                   selected_cols=','.join(selected_cols))

    return d, _decode_record


def predict_input_fn_builder(table, config):
    def input_fn():
        if FLAGS.task_type == C.TASK_TYPE_ITEM_EMBEDDING:
            dataset, decode = table_dataset_and_decode_builder(table, config, extra_fields=[('key', '')],
                                                               fea_sides=['item'])
        elif FLAGS.task_type == C.TASK_TYPE_USER_EMBEDDING:
            dataset, decode = table_dataset_and_decode_builder(table, config, extra_fields=[('key', '')],
                                                               fea_sides=['user'])

        dataset = dataset.repeat(1)
        dataset = dataset.map(decode)
        dataset = dataset.batch(batch_size=FLAGS.predict_batch_size)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element

    return input_fn


def eval_input_fn_builder(match_table, table, config):
    def input_fn():
        rank_dataset, rank_decode = table_dataset_and_decode_builder(table, config, extra_fields=[('label', 0)])

        rank_dataset = rank_dataset.repeat()
        rank_dataset = rank_dataset.shuffle(buffer_size=1000)
        rank_dataset = rank_dataset.map(rank_decode)
        rank_dataset = rank_dataset.batch(batch_size=FLAGS.train_batch_size)

        match_dataset, match_decode = table_dataset_and_decode_builder(match_table, config,
                                                                       extra_fields=[('label', 0), ('user_id', '')])
        match_dataset = match_dataset.repeat()
        match_dataset = match_dataset.shuffle(buffer_size=10000)
        match_dataset = match_dataset.map(match_decode)
        match_dataset = match_dataset.batch(batch_size=FLAGS.train_batch_size)

        neg_dataset, neg_decode = table_dataset_and_decode_builder(FLAGS.eval_neg_table, config, fea_sides=['item'],
                                                                   slice_id=0, slice_count=1, )
        neg_dataset = neg_dataset.repeat()
        neg_dataset = neg_dataset.shuffle(buffer_size=100000)
        neg_dataset = neg_dataset.map(neg_decode)
        neg_dataset = neg_dataset.batch(batch_size=10000)

        return tf.data.Dataset.zip(
            {'match_features': match_dataset, 'rank_features': rank_dataset, 'neg_features': neg_dataset})

    return input_fn


def input_fn_builder(match_table, table, config):
    def input_fn():
        rank_dataset, rank_decode = table_dataset_and_decode_builder(table, config, [('label', 0)])

        rank_dataset = rank_dataset.repeat()
        rank_dataset = rank_dataset.shuffle(buffer_size=1000)
        rank_dataset = rank_dataset.map(rank_decode)
        rank_dataset = rank_dataset.batch(batch_size=FLAGS.train_batch_size)

        match_dataset, match_decode = table_dataset_and_decode_builder(match_table, config, [('label', 0)])
        match_dataset = match_dataset.repeat()
        match_dataset = match_dataset.shuffle(buffer_size=10000)
        match_dataset = match_dataset.map(match_decode)
        match_dataset = match_dataset.batch(batch_size=FLAGS.train_batch_size)

        neg_dataset, neg_decode = table_dataset_and_decode_builder(FLAGS.train_neg_table, config, fea_sides=['item'],
                                                                   slice_id=0, slice_count=1, )
        neg_dataset = neg_dataset.repeat()
        neg_dataset = neg_dataset.shuffle(buffer_size=100000)
        neg_dataset = neg_dataset.map(neg_decode)
        neg_dataset = neg_dataset.batch(batch_size=100)

        return tf.data.Dataset.zip(
            {'match_features': match_dataset, 'rank_features': rank_dataset, 'neg_features': neg_dataset})

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

    def _match_loss_and_metric_v2(matchNet, config, features):
        match_features = features.get('match_features')
        neg_item_features = features.get('neg_features')

        user_emb_for_match = matchNet.user_embedding(match_features)
        item_emb_for_match = matchNet.item_embedding(match_features)

        neg_item_emb = matchNet.item_embedding(neg_item_features)

        pos_dis = tf.reduce_sum(user_emb_for_match * item_emb_for_match, axis=-1)
        neg_dis = tf.matmul(user_emb_for_match, neg_item_emb, transpose_b=True)

        dis = tf.concat([tf.expand_dims(pos_dis, -1), neg_dis], 1)

        prob = tf.nn.softmax(dis)
        match_loss = -tf.reduce_sum(tf.log(tf.slice(prob, [0, 0], [-1, 1]))) / FLAGS.train_batch_size

        return match_loss, tf.metrics.mean(metric_mrr(prob, 100))

    def _match_loss_and_metric_v3(matchNet, config, features):
        match_features = features.get('match_features')
        neg_item_features = features.get('neg_features')

        user_emb_for_match = matchNet.user_embedding(match_features)
        item_emb_for_match = matchNet.item_embedding(match_features)

        neg_item_emb = matchNet.item_embedding(neg_item_features)

        pos_dis = tf.reduce_sum(user_emb_for_match * item_emb_for_match, axis=-1)
        neg_dis = tf.matmul(user_emb_for_match, neg_item_emb, transpose_b=True)

        predictions = tf.concat([pos_dis, tf.reshape(neg_dis, [-1])], 0)
        labels = tf.concat(
            [tf.constant([1] * FLAGS.train_batch_size), tf.constant([0] * (FLAGS.train_batch_size * 100))], 0)

        match_loss = tf.losses.sigmoid_cross_entropy(labels, predictions)

        return match_loss, tf.metrics.auc(labels, tf.sigmoid(predictions))

    def _rank_loss_and_metric(matchNet, config, features):
        rank_features = features.get('rank_features')

        labels = rank_features['label']

        user_emb = matchNet.user_embedding(rank_features)
        content_emb = matchNet.item_embedding(rank_features)

        predictions = tf.reduce_sum(user_emb * content_emb, axis=1)
        rank_loss = tf.losses.sigmoid_cross_entropy(labels, predictions)

        return rank_loss, tf.metrics.auc(labels, tf.sigmoid(predictions))

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        matchNet = MatchNet(config)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # rank_loss, auc_metric = _rank_loss_and_metric(matchNet, config, features)
            match_loss, mrr_metric = _match_loss_and_metric(matchNet, config, features)
            loss = match_loss
            # loss += 0.1 * tf.reduce_mean(tf.reduce_sum(tf.square(user_emb), axis=1))
            # loss += 0.1 * tf.reduce_mean(tf.reduce_sum(tf.square(content_emb), axis=1))

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
            rank_loss, auc_metric = _rank_loss_and_metric(matchNet, config, features)
            match_loss, mrr_metric = _match_loss_and_metric(matchNet, config, features)

            user_embedding = matchNet.user_embedding(features.get('match_features'))
            item_embedding = matchNet.item_embedding(features.get('match_features'))
            neg_item_embedding = matchNet.item_embedding(features.get('neg_features'))

            user_id = features.get('match_features').get('user_id')
            pos_dis = tf.reduce_sum(user_embedding * item_embedding, axis=-1)

            unique_user_id, unique_user_embedding = unique_user_op(user_id, user_embedding)
            neg_dis = tf.matmul(unique_user_embedding, neg_item_embedding, transpose_b=True)

            hr, mrr = metric_cal_op(user_id, pos_dis, unique_user_id, neg_dis)

            loss = match_loss + rank_loss
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops={
                    "eval_loss": tf.metrics.mean(loss),
                    'rank_loss': tf.metrics.mean(rank_loss),
                    'match_loss': tf.metrics.mean(match_loss),
                    "auc": auc_metric,
                    'mrr': mrr_metric,
                    'mrr_v2': tf.metrics.mean(mrr),
                    'hr': tf.metrics.mean(hr),
                },
            )

        else:
            predictions = {}
            predictions['key'] = features.get('key')
            predict_type = FLAGS.task_type
            if predict_type == C.TASK_TYPE_USER_EMBEDDING:
                user_embedding = matchNet.user_embedding(features)
                user_embedding = tf.reduce_join(tf.as_string(user_embedding, precision=5), 1, separator=',')
                predictions['user_embedding'] = user_embedding
            elif predict_type == C.TASK_TYPE_ITEM_EMBEDDING:
                item_embedding = matchNet.item_embedding(features)
                item_embedding = tf.reduce_join(tf.as_string(item_embedding, precision=5), 1, separator=',')
                predictions['item_embedding'] = item_embedding
            elif predict_type == C.TASK_TYPE_PREDICT:
                pass

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def unique_user_op(user_id, user_embedding):
    def user_func(user_id, user_embedding):

        unique_user_id = []
        unique_user_embedding = []
        unique_set = set([])
        for i in range(len(user_id)):
            if user_id[i] not in unique_set:
                unique_user_id.append(user_id[i])
                unique_user_embedding.append(user_embedding[i])
                unique_set.add(user_id[i])

        return unique_user_id, unique_user_embedding

    y = tf.py_func(user_func, [user_id, user_embedding], [tf.string, tf.float32])
    y[0].set_shape((None))
    y[1].set_shape((None, user_embedding.get_shape()[1]))
    return y


def metric_cal_op(user_id, pos_dis, unique_user_id, neg_dis):
    def user_func(user_id, pos_dis, unique_user_id, neg_dis):
        user_id_2_samples = {}

        for i in range(len(unique_user_id)):
            samples = []
            for dis in neg_dis[i]:
                samples.append((dis, 0))

            user_id_2_samples[unique_user_id[i]] = samples

        for i in range(len(user_id)):
            user_id_2_samples[user_id[i]].append((pos_dis[i], 1))

        hit_size = 0
        mrr_sum = 0.0
        map_sum = 0.0
        for user_id, samples in user_id_2_samples.items():
            samples = sorted(samples, reverse=True)

            ap_sum = 0.0
            ap_size = 0
            is_first = True
            for i in range(min(len(samples), 800)):
                if samples[i][1] == 1:
                    hit_size += 1

                    if is_first:
                        mrr_sum += 1.0 / (i + 1)
                        is_first = False

        hr = hit_size * 1.0 / len(pos_dis)
        mrr = mrr_sum / len(user_id_2_samples)

        return np.float32(hr), np.float32(mrr)

    y = tf.py_func(user_func, [user_id, pos_dis, unique_user_id, neg_dis], [tf.float32, tf.float32])
    return y


def metric_mrr(predicts, neg_size):
    _, ranks = tf.nn.top_k(predicts, neg_size + 1)
    true_rank = tf.slice(tf.where(tf.equal(ranks, 0)), [0, 1], [FLAGS.train_batch_size, 1])
    mrr = tf.reduce_mean(1.0 / tf.to_float(true_rank + 1))
    return mrr


def test_input_fn(_input_fn):
    iterator = _input_fn().make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))


def main(_):
    config = {
        'embedding_size': 64,  # user and item final dim size
        'neg_sample_size': 5,
        'user': [
            # {
            #     'name': 'user_id',
            #     'size': 1000000,
            #     'value_index': 0,
            # },
            {
                'name': 'age',
                'size': 100,
                'need_hash': True,
                'type': 'one_hot'
            },
            {
                'name': 'gender',
                'size': 100,
                'need_hash': True,
                'type': 'one_hot'
            },
            {
                'name': 'purchase_total',
                'size': 100,
                'need_hash': True,
                'type': 'one_hot'  # one_hot,dense,
            },
            {
                'name': 'user_rootcates',
                'size': 10000,
                'need_hash': True,
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
                'need_hash': True,
                'type': 'one_hot'
            },
            {
                'name': 'cate_id',
                'size': 100000,
                'need_hash': True,
                'type': 'one_hot'
            },
        ],
    }

    model_fn = model_fn_builder(config)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)
    # distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        session_config=session_config,
        distribute=distribution,
        save_checkpoints_steps=50000,
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

    print(config)

    if FLAGS.task_type == C.TASK_TYPE_TRAIN:
        print(FLAGS.task_index, FLAGS.job_name, FLAGS.worker_count)
        if FLAGS.worker_count > 1:
            FLAGS.worker_count -= 1
        if FLAGS.task_index > 0:
            FLAGS.task_index -= 1
        train_input_fn = input_fn_builder(FLAGS.train_match_table, FLAGS.train_table, config)
        eval_input_fn = eval_input_fn_builder(FLAGS.eval_match_table, FLAGS.eval_table, config)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_max_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=60, start_delay_secs=30)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.task_type in (C.TASK_TYPE_USER_EMBEDDING, C.TASK_TYPE_ITEM_EMBEDDING):
        predict_input_fn = predict_input_fn_builder(FLAGS.input_table, config)

        writer = tf.python_io.TableWriter(FLAGS.output_table, slice_id=FLAGS.task_index)

        for result in estimator.predict(input_fn=predict_input_fn):
            if FLAGS.task_type == C.TASK_TYPE_USER_EMBEDDING:
                writer.write([result.get('key'), result.get('user_embedding')], [0, 1])
            elif FLAGS.task_type == C.TASK_TYPE_ITEM_EMBEDDING:
                writer.write([result.get('key'), result.get('item_embedding')], [0, 1])

        writer.close()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
