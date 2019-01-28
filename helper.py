import tensorflow as tf


class LogviewMetricWriter(object):
    def __init__(self):
        self.writer = tf.summary.MetricsWriter('.', flush_secs=10)

    def add_scalar(self, _metric_values, _step):
        for k, v in _metric_values.items():
            self.writer.add_scalar(k, v, _step)

    def __del__(self):
        self.writer.close()


class LogviewMetricHook(tf.train.SessionRunHook):
    def __init__(self, _metric_ops, _step_op, _logviewMetricWriter):
        self._step_op = _step_op
        self._metric_ops = {}
        self._logviewMetricWriter = _logviewMetricWriter
        for k, v in _metric_ops.items():
            self._metric_ops[k] = v[0]

    def end(self, session):
        if self._metric_ops is not None:
            _metric_values, _step = session.run([self._metric_ops, self._step_op])
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
