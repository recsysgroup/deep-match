import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# pai
flags.DEFINE_integer("task_index", 0, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("worker_count", 1, "")

# common
flags.DEFINE_string("input_table", None, "")
flags.DEFINE_string("output_table", None, "")
flags.DEFINE_string("tmp_dir", None, "")
flags.DEFINE_string("model_dir", None, "")
flags.DEFINE_integer("train_batch_size", 128, "")
flags.DEFINE_integer("train_max_step", 1000, "")
flags.DEFINE_boolean("do_train", True, "")
flags.DEFINE_boolean("do_predict", True, "")
flags.DEFINE_float("learning_rate", 1e-4, "")

# task
flags.DEFINE_string("input_emb_len", None, "")
flags.DEFINE_string("output_emb_len", None, "")


def input_fn_builder():
    pass


def model_fn_builder():
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument



        loss = None
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
                eval_metric_ops={"eval_loss": tf.metrics.mean(loss)}
            )

        else:
            predictions = None
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def main(_):
    print(FLAGS.task_index, FLAGS.job_name, FLAGS.worker_count)

    train_input_fn = input_fn_builder()
    eval_input_fn = input_fn_builder()
    predict_input_fn = input_fn_builder()

    model_fn = model_fn_builder()

    config = tf.ConfigProto(allow_soft_placement=True)
    distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        session_config=config,
        distribute=distribution,
        save_checkpoints_steps=1000,
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
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_max_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=10)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_predict:
        writer = tf.python_io.TableWriter(FLAGS.output_table, slice_id=FLAGS.task_index)
        result_iter = estimator.predict(input_fn=predict_input_fn)

        index = 0
        for result in result_iter:
            if index % 128 == 0:
                print(index, result['id'][0], result['emb'])
            writer.write([result['id'][0], result['emb']], [0, 1])
            index += 1

        tf.logging.info("writen done, size : {0}".format(index))
        writer.close()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
