import tensorflow as tf


class LogviewMetricWriter(object):
    def __init__(self, summury_dir):
        self.writer = tf.summary.MetricsWriter('.')
        self.tb_writer = tf.summary.MetricsWriter(summury_dir+'/my_evals')

    def add_scalar(self, _metric_values, _step):
        for k, v in _metric_values.items():
            self.writer.add_scalar(k, v, _step)
            self.tb_writer.add_scalar(k, v, _step)

    def __del__(self):
        self.writer.close()


class LogviewMetricHook(tf.train.SessionRunHook):
    def __init__(self, _metric_ops, _step_op, _logviewMetricWriter):
        self._step_op = _step_op
        self._metric_ops = {}
        self._logviewMetricWriter = _logviewMetricWriter
        for k, v in _metric_ops.items():
            self._metric_ops[k] = v

    def end(self, session):
        if self._metric_ops is not None:
            _metric_values, _step = session.run([self._metric_ops, self._step_op])
            for k, v in _metric_values.items():
                if type(v) is tuple:
                    _metric_values[k] = v[1]
                elif type(v) is dict:
                    for name, val in v.items():
                        _metric_values[name] = val
                    del _metric_values[k]
            self._logviewMetricWriter.add_scalar(_metric_values, _step)


class LogviewTrainHook(tf.train.SessionRunHook):
    def __init__(self, _loss_op, _learning_rate, _step_op, _logviewMetricWriter):
        self._loss_op = _loss_op
        self._learning_rate = _learning_rate
        self._step_op = _step_op
        self._logviewMetricWriter = _logviewMetricWriter
        self.cnt = 0

    def before_run(self, run_context):
        tensors = {
            'train_loss': self._loss_op,
            'learning_rate': self._learning_rate,
            'step': self._step_op
        }
        return tf.train.SessionRunArgs(tensors)

    def after_run(self,
                  run_context,
                  run_values):
        if self.cnt % 100 == 0:
            results = run_values.results
            _metrics = {
                'train_loss': results.get('train_loss'),
                'learning_rate': results.get('learning_rate'),
            }
            _step = results.get('step')
            self._logviewMetricWriter.add_scalar(_metrics, _step)

        self.cnt += 1


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
