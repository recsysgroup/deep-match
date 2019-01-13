# coding: utf-8

import tensorflow as tf
import json
import pprint
import os
import math

emb_col = tf.feature_column.embedding_column
ind_col = tf.feature_column.indicator_column
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
numeric_column = tf.feature_column.numeric_column
buck_col = tf.feature_column.bucketized_column

flags = tf.app.flags
flags.DEFINE_string("tables", "", "tables info")
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_string("checkpointDir", "", "outer model")
flags.DEFINE_string("mode", "train", "run mode")
flags.DEFINE_string("outputs", "", "output table")

FLAGS = flags.FLAGS

TRAIN = FLAGS.tables
TEST = FLAGS.tables
# , TEST = FLAGS.tables.split(
#    ',')  # "odps://etao_backend_dev/tables/DNN_TRAIN_S,odps://etao_backend_dev/tables/DNN_TEST_SS".split(',')

col_defaut = {
    'append_id': 0,
    'cnn_features': ",".join(["0"] * 4096),

}

USE_COLS = col_defaut.keys()
default = col_defaut.values()

assert len(USE_COLS) == len(default)
USE_COLS = ",".join(USE_COLS)
print(len(default))


# ------------------------------------------------------------------------------
def set_dist_env():
    # print(FLAGS.worker_hosts)
    # print(FLAGS.job_name,FLAGS.task_index )
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    # chief_hosts = worker_hosts[0:1]  # get first worker as chief
    # worker_hosts = worker_hosts[2:]  # the rest as worker
    # task_index = FLAGS.task_index
    if len(FLAGS.worker_hosts):
        cluster = {"chief": [worker_hosts[0]],
                   "ps": ps_hosts, "worker": worker_hosts[2:]}
        if FLAGS.job_name == "ps":
            os.environ['TF_CONFIG'] = json.dumps(
                {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})
        elif FLAGS.job_name == "worker":
            if FLAGS.task_index == 0:
                os.environ['TF_CONFIG'] = json.dumps(
                    {'cluster': cluster, 'task': {'type': "chief", 'index': 0}})
            elif FLAGS.task_index == 1:
                os.environ['TF_CONFIG'] = json.dumps(
                    {'cluster': cluster, 'task': {'type': "evaluator", 'index': 0}})
            else:
                os.environ['TF_CONFIG'] = json.dumps(
                    {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index - 2}})


def my_feature_columns():
    item__cnn_features = numeric_column(
        "item__cnn_features", shape=(4096,), default_value=0.0, )
    append_id = numeric_column("append_id", default_value=0, dtype=tf.int32)
    embedding_size = 32

    return {"deep": [], "append_id": [append_id], "subnet_feature": [item__cnn_features], "wide": [],
            "cross": []}


# assert sum(map(len, my_feature_columns().values())) == len(defaults) - 1
def input_fn(name, batch_size, slice_id=0, slice_count=1, repeat_count=None, ):
    # print("slice_count:%d, slice_id:%d" % (slice_count, slice_id))
    dataset = tf.data.TableRecordDataset([name], record_defaults=default, selected_cols=USE_COLS,
                                         slice_id=slice_id, slice_count=slice_count).prefetch(batch_size * 1000)
    # if repeat_count is None:
    #    dataset = dataset.shuffle(buffer_size=batch_size * 1000)

    dataset = dataset.repeat(repeat_count).batch(batch_size)

    tensor_list = dataset.make_one_shot_iterator().get_next()

    append_id, item__cnn_features = tensor_list

    # unique_key = tf.as_string(unique_key)
    a = tf.string_split(item__cnn_features, ',')
    b = tf.sparse_tensor_to_dense(a, default_value='0.0')
    features = tf.string_to_number(b)

    return {"append_id": append_id,
            "item__cnn_features": features}, None


def my_model(features, labels, mode, params):
    train_flag = (mode == tf.estimator.ModeKeys.TRAIN)
    drop_prob = params["drop_prob"]
    l2_reg = params["l2_reg"]
    init_std = params["init_std"]
    pos_weights = params["pos_weight"]
    use_wide = params["use_wide"]
    use_subnet = params["use_subnet"]
    subnet_units = params["subnet_units"]

    deep_optimizer = params["deep_optimizer"]
    deep_lr = params["deep_lr"]

    if use_subnet and len(subnet_units) == 0:
        raise ValueError()

    # ----------------------------#
    input_feature_column_dict = params['feature_column_dict']
    append_id = tf.feature_column.input_layer(features, input_feature_column_dict["append_id"], trainable=False, )
    # -------------auto encoder-------------
    cnn_input = tf.feature_column.input_layer(
        features, input_feature_column_dict['subnet_feature'])
    encoder_input = cnn_input
    for units in [1024, 256]:
        encoder_input = tf.layers.dense(encoder_input, units, activation=tf.nn.elu,
                                        kernel_initializer=tf.glorot_uniform_initializer(), name="encoder" + str(units))
    encoder_input = tf.layers.dense(encoder_input, 32, activation=None,
                                    kernel_initializer=tf.glorot_uniform_initializer(), name="encoder" + str(32))

    decoder_output = encoder_input
    for units in [256, 1024]:
        decoder_output = tf.layers.dense(decoder_output, units, activation=tf.nn.relu,
                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                         name="decoder" + str(units))
    decoder_output = tf.layers.dense(decoder_output, 4096, activation=None,
                                     kernel_initializer=tf.glorot_uniform_initializer(), name="decoder" + str(4096))

    loss = tf.losses.mean_squared_error(cnn_input, decoder_output)

    if deep_optimizer == 'Adam':
        optimizer = tf.train.AdamAsyncOptimizer(
            learning_rate=deep_lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif deep_optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=deep_lr, initial_accumulator_value=0.1)
    elif deep_optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=deep_lr, momentum=0.95)
    elif deep_optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            deep_lr, l1_regularization_strength=0.5, l2_regularization_strength=1.0)
    else:
        raise ValueError()

    deep_opt = optimizer  # tf.train.AdagradOptimizer(0.001)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'append_id': append_id,
            "encoder_input": cnn_input,
            'encoder_output': encoder_input,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    metrics = {}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    # deep_train_op_loss = tf.contrib.training.create_train_op(loss, deep_opt, global_step=tf.train.get_global_step(),summarize_gradients=True)
    deep_train_op = deep_opt.minimize(loss, global_step=tf.train.get_global_step(), )
    # optimizer = tf.train.AdamAsyncOptimizer(learning_rate=0.001)

    train_op = deep_train_op
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(args):
    # print("job name = %s" % FLAGS.job_name)
    # print("task index = %d" % FLAGS.task_index)

    # ---------------------------------------------------------------------------------------------------------------------------
    # drop_rate = FLAGS.drop_rate
    my_params = {"subnet_units": [32, ], "units": [256, 32], "use_wide": True,
                 "use_deep": True,
                 "use_cross": False, "use_subnet": True,
                 "drop_prob": 0.5, "l2_reg": 0.0, "init_std": 1e-4, "pos_weight": 1.1, "deep_lr": 0.005,
                 "deep_optimizer": "Adagrad"}
    if my_params['use_subnet'] and not my_params['use_deep']:
        raise ValueError("use_deep must be true if use_subnet is true")
    pprint.pprint(my_params)
    my_params['feature_column_dict'] = my_feature_columns()

    run_config = tf.estimator.RunConfig(
        tf_random_seed=1024, save_checkpoints_secs=30, keep_checkpoint_max=5)
    DIR = FLAGS.checkpointDir
    # DIR = FLAGS.buckets
    classifier = tf.estimator.Estimator(
        model_fn=my_model, model_dir=DIR, params=my_params, config=run_config)

    if FLAGS.mode == "train":
        print("start training")
        train_total_num, test_total_num = 85503, 85503
        batch_size = 256
        test_batch_size = 2 ** 13
        steps_per_epoch = train_total_num / batch_size
        # steps_per_epoch = 1
        train_epoch = 200
        test_steps_per_epoch = test_total_num / test_batch_size

        print("train steps per epoch", steps_per_epoch,
              "test steps per epoch", test_steps_per_epoch)
        # 8min eval all
        worker_count = len(FLAGS.worker_hosts.split(','))
        if FLAGS.task_index is None:
            task_index = 0
        else:
            task_index = FLAGS.task_index
        print("task_index", task_index, "worker_count", worker_count)

        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
            TRAIN, batch_size, task_index, worker_count), max_steps=steps_per_epoch * train_epoch)  # steps_per_epoch
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(TEST, test_batch_size, task_index, worker_count, 1),
                                          steps=None,
                                          start_delay_secs=30,
                                          throttle_secs=30)

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
        print("done")
    elif FLAGS.mode == "predict":
        print("start predicting")
        test_batch_size = 2 ** 13
        results = classifier.predict(input_fn=lambda: input_fn(TEST, test_batch_size, 0, 1, 1), predict_keys=None)
        values_list = []
        for ans in results:
            # input_row = ','.join(map(lambda x:"%.6f"%x,list(ans["encoder_input"])))
            append_id = int(ans['append_id'][0])
            write_row = ','.join(map(str, list(ans['encoder_output'])))
            values_list.append((append_id, write_row))
        print("records len", len(values_list))
        writer = tf.python_io.TableWriter(FLAGS.outputs)
        records = writer.write(values_list, indices=[0, 1])
        writer.close()
        print("prediction done")


if __name__ == "__main__":
    set_dist_env()

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
